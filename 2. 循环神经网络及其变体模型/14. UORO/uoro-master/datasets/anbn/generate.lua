require 'options'

local opt = lapp[[
-o, --outfile          (default 'data')        radical of the output file
-n, --samples          (default 1e7)           minimum number of characters
-m, --min              (default 1)             minimum number of characters between save
-M, --max              (default 32)            maximum number of characters between save
]]

local outfile = options.appendOptions('data', opt, {outfile=true, samples=true})..'.dat'
local outt7 = options.appendOptions('data', opt, {outfile=true, samples=true})..'.t7'
local vocabt7 = options.appendOptions('vocab', opt, {outfile=true, samples=true})..'.t7'

io.output(outfile)

local n_sample = 0

local range = 0

while n_sample < opt.samples do
    range = torch.Tensor(1):random(opt.min, opt.max)[1]
    for i=1, range do
        io.write('a')
    end
    io.write('\n')
    for i=1, range do
        io.write('b')
    end
    io.write('\n')
    n_sample = n_sample + 2 * range + 2
end
io.flush()

io.output():close()

options.toT7(outfile, outt7, vocabt7)
