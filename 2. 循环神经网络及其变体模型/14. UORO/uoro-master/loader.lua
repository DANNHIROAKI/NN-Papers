loader = {}

function loader.textLoader(opt)
    local raw_data     = torch.load(opt.datafile)
    local size         = raw_data:size(1)
    local batch_number = math.floor(size / opt.batch)

    size = batch_number * opt.batch
    raw_data = raw_data[{{1, size}}]

    local x_data = raw_data
    local y_data = raw_data:clone()
    y_data[{{1, size-1}}]:copy(x_data[{{2, size}}])
    y_data[size] = x_data[1]
    x_data = x_data:view(batch_number, opt.batch)
    y_data = y_data:view(batch_number, opt.batch)

    local character_number = x_data:max()
    local index = 1
    
    local function batch()
        local x_batch = x_data[(index - 1)%batch_number + 1]
        local y_batch = y_data[(index - 1)%batch_number + 1]
        index = index + 1
        return x_batch, y_batch
    end

    return x_data, y_data, character_number, batch
end
