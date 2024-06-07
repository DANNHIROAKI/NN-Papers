require 'nngraph'
require 'optim'
require 'modules'
require 'options'
require 'loader'

local UORO = require 'uoro'

local opt = lapp[[
-c, --cuda                                                       specify use of gpus
-m, --model    (default 'GRU')                                   specify model
-t, --threads  (default 1)                                       specify number of threads
-s, --save     (default 'logs')                                  specify save file
-r, --rate     (default 1e-3)                                    specify learning rate
-d, --datafile (default 'datasets/anbn/data-min-1-max-32.t7')    specify data file
-f, --modelfile(default '')                                      specify model file
-M, --max      (default 1e7)                                     specify maximum number of iterations
-b, --batch    (default 1)                                       specify batch size
-T, --truncate (default 1)                                       specify truncation parameter
-o, --off                                                        if specified then just BPTT
-w, --width    (default 'small')                                 specify network width
-a, --alpha    (default 3e-2)                                    specify learning time lag
]]


torch.manualSeed(0)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> number of threads used ' .. torch.getnumthreads())
-- --

-- cuda import
if opt.cuda then
    require 'cunn'
    require 'cudnn'
end
-- --

-- logging
local savefile = options.appendOptions(opt.save..'/results', opt, {cuda=true, threads=true, save=true, modelfile=true, momentum=true, forget=true, vocabfile=true, testfile=true, datafile=true, max=true})
local logger = optim.Logger(savefile)
-- --

-- data preparation
local x_data, y_data, v_size, batch = loader.textLoader{datafile=opt.datafile, batch=opt.batch}
-- --

-- define sizes
local size = {}
size.o = v_size
size.x = size.o
-- --

-- models
local model
local infos = {x={1}, o={size.o}}

local size_table = {big=1024,small=64,tiny=8}
size.s = size_table[opt.width]

if opt.model == 'GRU' then
    local i = nn.Identity()()
    local h = nn.Identity()()
    local r = nn.Sigmoid()(nn.CAddTable(){
        nn.Embedding(size.x, size.s)(i),
        nn.Linear(size.s, size.s)(h)
    })
    local z = nn.Sigmoid()(nn.CAddTable(){
        nn.Embedding(size.x, size.s)(i),
        nn.Linear(size.s, size.s)(h)
    })
    local h_tilde = nn.Tanh()(nn.CAddTable(){
        nn.Embedding(size.x, size.s)(i),
        nn.Linear(size.s, size.s)(nn.CMulTable(){r,h})

    })

    local h_next = nn.CAddTable(){
        nn.CMulTable(){z, h_tilde},
        nn.CMulTable(){h, nn.AddConstant(1)(nn.MulConstant(-1)(z))}
    }
    local out = nn.Linear(size.s, size.o)(h_next)

    infos.s = {size.s}

    model = nn.gModule({i, h}, {out, h_next})
elseif opt.model == 'LSTM' then
    local i = nn.Identity()()
    local embed = nn.Embedding(size.x, size.s)(i)
    local h = nn.Identity()()
    local c = nn.Identity()()

    local function input_sum(input, hidden, bias)
        bias = bias or 0
        return nn.AddConstant(bias)(
        nn.CAddTable(){
            nn.Linear(size.s, size.s)(input),
            nn.Linear(size.s, size.s)(hidden)
        })
    end

    local forget_gate = nn.Sigmoid()(input_sum(embed, h, -0.3))
    local input_gate = nn.Sigmoid()(input_sum(embed, h))
    local output_gate = nn.Sigmoid()(input_sum(embed, h))
    local c_tilde = nn.Tanh()(input_sum(embed, h))
    local c_next = nn.CAddTable(){
        nn.CMulTable(){
            forget_gate,
            c
        },
        nn.CMulTable(){
            input_gate,
            c_tilde
        }
    }


    local h_next = nn.CMulTable(){
        output_gate,
        nn.Tanh()(c_next)
    }

    local out = nn.Linear(size.s, size.o)(h_next)

    infos.s = {size.s, size.s}

    model = nn.gModule({i, c, h}, {out, c_next, h_next})
elseif opt.model == 'RNN' then
    local i = nn.Identity()()
    local h = nn.Identity()()
    local h_next = nn.Tanh()(
        nn.CAddTable(){
            nn.Linear(size.s, size.s)(h),
            nn.Embedding(size.x, size.s)(i)
        })

    local out = nn.Linear(size.s, size.o)(h_next)

    infos.s = {size.s}

    model = nn.gModule({i, h}, {out, h_next})
end

local criterion = nn.CrossEntropyCriterion()

if opt.cuda then 
    model = model:cuda()
    criterion = criterion:cuda()
end
-- --

local parameter, gradient = model:getParameters()

local uoro = UORO:new{model=model, infos=infos, theta=parameter, g=gradient,
                      criterion=criterion, T=opt.truncate, off=opt.off, 
                      b_size=opt.batch}
if opt.cuda then
    uoro = uoro:cuda()
end

local function feval(param)
    if param ~= parameter then
        parameter:copy(param)
    end

    gradient:zero()

    local inputs = {}
    local targets = {}
    for t=1, opt.truncate do
        local c_input, c_target = batch()
        inputs[t] = c_input:clone()
        targets[t] = c_target:clone()
    end

    local loss, gradient = uoro:forward(inputs, targets)

    return loss, gradient
end

local optimState = {}

local lastprint=1

while uoro.epoch < opt.max do
    optimState.learningRate = opt.rate / (1 + opt.alpha * math.sqrt(uoro.epoch)) 
    _, loss = optim.adam(feval, parameter, optimState)

    recentLoss = recentLoss and recentLoss*(1-1/math.sqrt(uoro.epoch))+loss[1]/math.sqrt(uoro.epoch)/uoro.T or loss[1] / uoro.T
    cumulativeLoss = cumulativeLoss and cumulativeLoss + loss[1] or loss[1]

    if uoro.epoch > lastprint then
        lastprint = lastprint*1.01
        logger:add{
            ['epoch'] = uoro.epoch-1,
            ['cumulative loss'] = cumulativeLoss / uoro.epoch / math.log(2),
            ['recent loss'] = recentLoss / math.log(2),
        }

        formatter = '%8.0f %5.4f %5.4f'
        print(string.format(formatter, uoro.epoch, recentLoss/math.log(2),
        cumulativeLoss / uoro.epoch / math.log(2)))
    end
end
