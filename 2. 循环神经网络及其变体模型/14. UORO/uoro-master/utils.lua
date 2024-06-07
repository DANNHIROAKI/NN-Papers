local utils = {}

function utils.cloneNetwork(net, T)
    -- return T copies of the initial network net
    -- sharing the parameters between the copies
    local clones = {}

    -- retrieve parameters
    local params, gradParams
    if net.parameters then
        params, gradParams = net:parameters()
        if params == nil then
            params = {}
        end
    end

    local paramsNoGrad
    if net.parametersNoGrad then
        paramsNoGrad = net:parametersNoGrad()
    end

    -- store the structure of the network
    -- in a virtual file
    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net)

    -- copy
    for t = 1, T do
        -- We needo to use a new reader for each clone.
        -- We don't want to use the pointers to already read objects.
        local reader = torch.MemoryFile(mem:storage(), "r"):binary()
        local clone = reader:readObject()
        reader:close()

        -- the clone parameters must point to the same
        -- address locations as the original net parameters
        if net.parameters then
            local cloneParams, cloneGradParams = clone:parameters()
            local cloneParamsNoGrad
            for i=1,#params do
                cloneParams[i]:set(params[i])
                cloneGradParams[i]:set(gradParams[i])
            end
            if paramsNoGrad then
                cloneParamsNoGrad = clone:parametersNoGrad()
                for i=1,#paramsNoGrad do
                    cloneParamsNoGrad[i]:set(paramsNoGrad[i])
                end
            end
        end

        clones[t] = clone
        collectgarbage()
    end

    mem:close()
    return clones
end

function utils.randomFromArray(a)
    local rand = torch.Tensor(1):uniform()
    local cumSum = 0
    for i=1,a:nElement() do
       cumSum = cumSum + a[i] 
       if cumSum > rand[1] then
           return i
       end
    end
    return a:nElement()
end
return utils
