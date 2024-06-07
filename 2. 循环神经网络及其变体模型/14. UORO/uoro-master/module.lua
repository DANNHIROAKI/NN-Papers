-- This file defines some modifications to the Module class
-- mostly defining forward propagation


-- if propagateGrad is defined in the submodule, use it
-- otherwise fallback to numerical gradient
function nn.Module:forwardDiff(input, gradInput)
    -- __init cannot be overloaded, self.gradOutput is defined here
    self.gradOutput = self.gradOutput or torch.Tensor()
    -- --
    
    if not self.propagate then
        -- numerical gradient constant
        local epsilon = 1e-5
        -- --
        -- normalize gradInput and store norm
        local norm = 0
        if type(gradInput) == 'table' then
            for i=1,#gradInput do
                norm = norm + gradInput[i]:norm() ^ 2
            end
            norm = math.sqrt(norm)
            if norm ~= 0 then
                for i=1,#gradInput do
                    gradInput[i]:div(norm)
                end
            end
        else
            norm = gradInput:norm()
            if norm ~= 0 then
                gradInput:div(norm)
            end
        end
        -- --

        -- resize gradOutput if necessary
        if type(self.output) == 'table' and type(self.gradOutput) ~= 'table' then
            self.gradOutput = {}
            for index=1,#self.output do
                self.gradOutput[index] = self.output[index]:clone()
            end
        elseif type(self.gradOutput) ~= 'table' then
            if self.gradOutput:nElement() ~= self.output:nElement() then
                self.gradOutput:resizeAs(self.output)
            end
        end
        -- --


        -- create a buffer for the input to avoid
        -- numerical complications
        if not self.buffer and type(input) == "table" then
            self.buffer = {}
            for i=1,#input do
                self.buffer[i] = input[i]:clone()
            end
        else
            self.buffer = self.buffer or input:clone()
        end
        -- --
        -- copy the input
        if type(input) == "table" then
            for i=1,#input do
                self.buffer[i]:copy(input[i])
            end
        else
            self.buffer:copy(input)
        end
        -- --

        -- add epsilon * gradInput to the input
        if type(input) == "table" then
            for index=1,#input do
                -- particular case when gradInput is 0 (allows for Embedding)
                if gradInput[index]:norm() ~= 0 then
                    input[index]:add(epsilon, gradInput[index])
                end
                -- --
            end
        else
            if gradInput:norm() ~= 0 then
                input:add(epsilon, gradInput)
            end
        end
        -- --
        -- forward and copy to gradOutput
        self:forward(input)
        if type(self.gradOutput) == 'table' then
            for index=1,#self.output do
                self.gradOutput[index]:copy(self.output[index])
            end
        else
            self.gradOutput:copy(self.output)
        end
        -- --

        -- restore input and forward
        if type(input) == "table" then
            for index=1,#input do
                if gradInput[index]:norm() ~= 0 then
                    input[index]:copy(self.buffer[index])
                end
            end
        else
            if gradInput:norm() ~= 0 then
                input:copy(self.buffer)
            end
        end
        -- --
        -- forward and csub to gradOutput
        self:forward(input)
        if type(self.gradOutput) == 'table' then
            for index=1,#self.output do
                self.gradOutput[index]:csub(self.output[index])
                self.gradOutput[index]:div(epsilon)
            end
        else
            self.gradOutput:csub(self.output)
            self.gradOutput:div(epsilon)
        end
        -- --

        -- takes norm into account
        if norm ~= 0 then
            if type(gradInput) == 'table' then
                for i=1, #gradInput do
                    gradInput[i]:mul(norm)
                end
            else
                gradInput:mul(norm)
            end
            if type(self.gradOutput) == 'table' then
                for i=1, #self.gradOutput do
                    self.gradOutput[i]:mul(norm)
                end
            else
                self.gradOutput:mul(norm)
            end
        end
        -- --
    else
        self:propagate(input, gradInput)
    end
    return self.gradOutput
end
-- --
