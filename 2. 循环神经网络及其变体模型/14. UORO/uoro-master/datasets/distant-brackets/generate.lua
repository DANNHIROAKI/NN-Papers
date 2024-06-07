require 'options'

local opt = lapp[[
-o, --outfile          (default 'data')        radical of the output file
-n, --samples          (default 1e7)           minimum number of characters
-m, --min              (default 5)             minimum number of characters between save
-M, --max              (default 5)             maximum number of characters between save
-s, --save             (default 1)             number of characters to save
-a, --alphabet         (default 10)            number of letters in alphabet
]]

local outfile = options.appendOptions('data', opt, {outfile=true, samples=true})..'.dat'
local outt7 = options.appendOptions('data', opt, {outfile=true, samples=true})..'.t7'
local vocabt7 = options.appendOptions('vocab', opt, {outfile=true, samples=true})..'.t7'

io.output(outfile)

local n_sample = 0

local reserved = {[91]=true,[93]=true}
local delimiters = {left='[',right=']'}
local rlb = 33 -- random lower bound
local rub = 33 + opt.alphabet - 1 -- random upper bound
local random = 0
local range = 0
local stored = {}

while n_sample < opt.samples do
    io.write(delimiters.left)
    for i=1, opt.save do
        repeat
            random = torch.Tensor(1):random(rlb, rub)[1]
            io.write(string.char(random))
        until not reserved[random]
        stored[i] = random
    end
    io.write(delimiters.right)
    range = torch.Tensor(1):random(opt.min, opt.max)[1]
    for i=1, range do
        repeat
            random = torch.Tensor(1):random(rlb, rub)[1]
            io.write(string.char(random))
        until not reserved[random]
    end
    io.write(delimiters.left)
    for i=1, opt.save do
        io.write(string.char(stored[i]))
    end
    io.write(delimiters.right)
    io.write('\n')
    n_sample = n_sample + 2 * (opt.save+2) + range + 1
end

io.output():close()

options.toT7(outfile, outt7, vocabt7)
