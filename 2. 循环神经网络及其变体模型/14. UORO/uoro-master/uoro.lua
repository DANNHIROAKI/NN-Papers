require 'module'

local utils = require 'utils'

local UORO = {}
UORO.__index = UORO

function UORO:new(opt)
    local instance = {}

    instance.b_size    = opt.b_size or 1
    instance.T         = opt.T or 1
    instance.off       = opt.off -- TBPTT if off instance.allocated = false
    instance.epoch     = 1

    local function needed(p)
        assert(opt[p], p .. ' required')
        instance[p] = opt[p]
    end

    needed('theta')
    needed('g')
    needed('model')
    needed('criterion')
    needed('infos')

    assert(opt.infos.x and opt.infos.s and opt.infos.o,
           'input, state and output infos required')
    

    local x_size     = 0
    local s_size     = 0
    local o_size     = 0
    local theta_size = instance.theta:nElement()

    for i=1,#opt.infos.x do
        x_size  = x_size + opt.infos.x[i]
    end

    for i=1,#opt.infos.s do
        s_size  = s_size + opt.infos.s[i]
    end

    for i=1,#opt.infos.o do
        o_size = o_size + opt.infos.o[i]
    end

    instance.storage = torch.Tensor(((instance.T + 1)*x_size + o_size + 4*s_size)*instance.b_size + 2*theta_size):zero()

    instance.x              = {}
    instance.s              = {}
    instance.sbar           = {}
    instance.zero_x         = {}
    instance.zero_s         = {}
    instance.zero_o         = {}
    instance.nu             = {}


    instance.clones = {}

    instance.clones.model     = utils.cloneNetwork(instance.model, instance.T)
    instance.clones.criterion = utils.cloneNetwork(instance.criterion, instance.T)

    return setmetatable(instance, self)
end

function UORO:forward(x, o_hat)
    assert(type(x)  == 'table', 'x should be a table of ' .. self.T .. ' elements.')
    assert(type(o_hat) == 'table', 'o_hat should be a table of ' .. self.T .. ' elements.')

    self.o_hat = {}
    if not self.allocated then
        self:allocate()
    end

    for t=1, self.T do
        if type(x[t]) ~= 'table' then
            x[t] = {x[t]}
        end
        if type(o_hat[t]) ~= 'table' then
            o_hat[t] = {o_hat[t]}
        end
        -- copy inputs and targets into specific locations
        for i=1, #self.x[t] do
            -- correct a small misbehavior on Embeddings
            self.x[t][i] = self.x[t][i]:viewAs(x[t][i])
        end
        self.o_hat[t] = {} 
        for i=1, #o_hat[t] do
            self.o_hat[t][i] = o_hat[t][i]:clone()
        end
    end

    for t=1, self.T do
        for i=1, #self.x[t] do
            self.x[t][i]:copy(x[t][i])
        end
        for i=1, #self.o_hat[t] do
            self.o_hat[t][i]:copy(o_hat[t][i])
        end
    end
    -- --

    -- forward pass
    local currents = {}
    self.ss = {[0]=self.s}
    self.o = {}
    local loss = 0
    for t=1, self.T do
        currents[t] = {}
        for i=1, #x do
            currents[t][i] = self.x[t][i]
        end
        for i=1, #self.s do
            currents[t][#currents[t] + 1] = self.ss[t-1][i]
        end

        local forwarded = self.clones.model[t]:forward(currents[t])

        self.o[t] = {}
        self.ss[t] = {}
        for i=1, #self.infos.o do
            self.o[t][i] = forwarded[i]
        end
        for i=#self.infos.o + 1, #self.infos.o + #self.s do 
            self.ss[t][#self.ss[t] + 1] = forwarded[i] 
        end
        -- --
        if #self.o[t] == 1 then
            loss = loss + self.clones.criterion[t]:forward(self.o[t][1], self.o_hat[t][1])
        else
            loss = loss + self.clones.criterion[t]:forward(self.o[t], self.o_hat[t])
        end
    end

    -- differential pass
    -- compute gradient estimate
    self.dldo = {}
    self.toBackwards = {}
    self.delta_s = {[self.T]=self.zero_s}
    for t=self.T, 1, -1 do
        self.toBackwards[t] = {}
        self.delta_s[t-1] = {}
        if #self.o[t] == 1 then
            self.dldo[t] = self.clones.criterion[t]:backward(self.o[t][1], self.o_hat[t][1])
            self.toBackwards[t][1] = self.dldo[t]
        else
            self.dldo = self.clones.criterion[t]:backward(self.o[t], self.o_hat[t])
            for i=1, #self.o[t] do
                self.toBackwards[t][i] = self.dldo[t][i]
            end
        end
        for i=1, #self.s do
            self.toBackwards[t][#self.o[t] + i] = self.delta_s[t][i]
        end
        local backwarded = self.clones.model[t]:backward(currents[t], self.toBackwards[t])
        for i=1, #self.s do
            self.delta_s[t-1][i] = backwarded[#self.x[t] + i]
        end
    end
    if not self.off then
        for i=1, #self.s do
            self.g:add(self.delta_s[0][i]:dot(self.sbar[i]), self.thetabar)
        end

        -- compute gradient and put it into a buffer
        self.theta_buffer:copy(self.g)
        -- --
        
        self.g:zero()
        self.transsbar = {}
        local toForward = {}
        for i=1, #x[1] do
            toForward[i] = self.zero_x[i]
        end
        for i=1, #self.s do
            toForward[#x[1] + i] = self.sbar[i]
        end
        for t=1, self.T do
            local forwardedDifferential = self.clones.model[t]:forwardDiff(currents[t], toForward) 
            for i=1, #self.s do
                toForward[#self.x[1] + i] = forwardedDifferential[#self.o[t]+i]
            end
        end
        for i=1, #self.s do
            self.transsbar[i] = toForward[#self.x[1]+i]
        end
        for i=1, #self.s do
            self.nu[i]:bernoulli():mul(2):add(-1)
        end
        self.delta_s[self.T] = self.nu
        for t=self.T, 1, -1 do
            for i=1, #self.o[t] do
                self.toBackwards[t][i] = self.zero_o[i]
            end
            for i=1, #self.s do
                self.toBackwards[t][#self.o[t] + i] = self.delta_s[t][i]
            end
            local backwarded = self.clones.model[t]:backward(currents[t], self.toBackwards[t])
            for i=1, #self.s do
                self.delta_s[t-1][i] = backwarded[#self.x[t] + i]
            end
        end

        -- compute norms
        nnu = 0
        for i=1, #self.s do
            nnu = nnu + self.nu[i]:dot(self.nu[i])
        end
        nnu = math.sqrt(nnu)
        ntranssbar = 0
        for i=1, #self.s do
            ntranssbar = ntranssbar + self.transsbar[i]:dot(self.transsbar[i])
        end
        ntranssbar = math.sqrt(ntranssbar)
        -- --

        local epsilon = 1e-5
        local rho0 = math.sqrt((self.thetabar:norm() + epsilon)/(ntranssbar + epsilon))
        local rho1 = math.sqrt((self.g:norm() + epsilon)/(nnu + epsilon))
        for i=1, #self.s do
            self.sbar[i]:copy(self.transsbar[i]):mul(rho0):add(rho1, self.nu[i])
        end
        self.thetabar:div(rho0):add(1/rho1, self.g)
        -- --
        -- restore gradient
        self.g:copy(self.theta_buffer)
        -- --
    end

    -- update state
    for i=1, #self.s do
        self.s[i]:copy(self.ss[self.T][i])
    end
    -- --

    self.epoch = self.epoch + self.T

    return loss, self.g
end

-- works inplace, but accepts common syntax
-- does not cuda parameters and gradient nor model
function UORO:cuda()
    self.storage = self.storage:cuda()
    self.storage:zero()
    return self
end

-- utility function to allocate input, state and output 
function UORO:allocate()
    self.allocated = true
    local index = 1

    local function allocateSpace(i, t, storage, dim)
        local size = 1
        for j=1, dim:size(1) do
            size = size * dim[j]
        end
        t[i] = storage[{{index, index + size - 1}}]:view(dim)
        index = index + size
    end

    for t=1, self.T do
        for i=1,#self.infos.x do
            local dim = torch.LongStorage(2)
            dim[1] = self.b_size 
            dim[2] = self.infos.x[i]
            self.x[t] = {}
            allocateSpace(i, self.x[t], self.storage, dim)
        end
    end
    for i=1,#self.infos.x do
        local dim = torch.LongStorage(2)
        dim[1] = self.b_size 
        dim[2] = self.infos.x[i]
        allocateSpace(i, self.zero_x, self.storage, dim)
    end
    for i=1,#self.infos.s do
        local dim = torch.LongStorage(2)
        dim[1] = self.b_size 
        dim[2] = self.infos.s[i]
        allocateSpace(i, self.s, self.storage, dim)
        allocateSpace(i, self.zero_s, self.storage, dim)
        allocateSpace(i, self.nu, self.storage, dim)
        allocateSpace(i, self.sbar, self.storage, dim)
    end
    for i=1,#self.infos.o do
        local dim = torch.LongStorage(2)
        dim[1] = self.b_size 
        dim[2] = self.infos.o[i]
        allocateSpace(i, self.zero_o, self.storage, dim)
    end
    local dim = self.theta:size()
    self.thetabar= self.storage[{{index, index + self.theta:size(1) - 1}}]:viewAs(self.theta)
    index = index + self.theta:size(1)
    self.theta_buffer= self.storage[{{index, index + self.theta:size(1) - 1}}]:viewAs(self.theta)
    index = index + self.theta:size(1)
end

return UORO
