options = {}

function options.appendOptions(text, options, ignore)
    for o, v in pairs(options) do
        if type(v) == 'number' then
            v = string.format('%3.2e',v)
        end
        if not ignore[o] then
            text = text..'-'..o..'-'..tostring(v)
        end
    end
    return text
end

function options.parseOptions(text, default)
    local results = {}
    for k, v in pairs(default) do
        results[k] = v
    end
    for pattern in string.gmatch(text, '%-.-%-%d*') do
        local i1, i2 = string.find(pattern, '%a+')
        local key = string.sub(pattern, i1, i2)
        i1, i2 = string.find(pattern, '%d+')
        local value = string.sub(pattern, i1, i2)
        value = tonumber(value) or value
        results[key] = value
    end
    return results
end

function options.toT7(infile, outfile, vocabfile)
    local f = torch.DiskFile(infile)
    local rawdata = f:readString('*a')
    f:close()

    local unordered = {}
    for char in rawdata:gmatch'.' do
        if not unordered[char] then unordered[char] = true end
    end

    local ordered = {}
    for char in pairs(unordered) do ordered[#ordered+1]=char end
    table.sort(ordered)

    local vocab_mapping = {}
    for i, char in ipairs(ordered) do
        vocab_mapping[char] = i
    end

    local data = torch.ByteTensor(#rawdata)
    for i=1, #rawdata do
        data[i] = vocab_mapping[rawdata:sub(i, i)]
    end

    torch.save(outfile, data)
    torch.save(vocabfile, vocab_mapping)
end

return options
