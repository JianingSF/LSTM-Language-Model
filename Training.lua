require "env"
require 'nn'      
require 'nngraph'
require 'io'
require 'cunn'
require 'optim'
include 'debugger.lua'
include 'CriterionStock.lua'
include 'utils.lua'

local npy4th = require 'npy4th'

function lstm(i, prev_c, prev_h, rnn_size)
  function new_input_sum()
    local i2h            = nn.Linear(rnn_size, rnn_size)
    local h2h            = nn.Linear(rnn_size, rnn_size)
    return nn.CAddTable()({i2h(i), h2h(prev_h)})
  end
  local in_gate          = nn.Sigmoid()(new_input_sum())
  local forget_gate      = nn.Sigmoid()(new_input_sum())
  local in_gate2         = nn.Tanh()(new_input_sum())
  local next_c           = nn.CAddTable()({
    nn.CMulTable()({forget_gate, prev_c}),
    nn.CMulTable()({in_gate,     in_gate2})
  })
  local out_gate         = nn.Sigmoid()(new_input_sum())
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
  return next_c, next_h
end

function create_network(layers, wordEmbedding_size, rnn_size,input_size,output_size,vocSize, useCuda)
  local useCuda = 1
  print ('useCuda ' ..useCuda.. '\n')
  local x                = nn.Identity()()
  --local y                = nn.Identity()()
  local prev_s           = nn.Identity()()
  local i                = {[0] = nn.Linear(wordEmbedding_size, rnn_size)(nn.LookupTable(input_size, wordEmbedding_size)(x))}
  --local i                = {[0] = nn.Linear(input_size, rnn_size)(x) } 
  local next_s           = {}
  local splitted         = { prev_s:split(2 * layers) }
  for layer_idx = 1, layers do
    local prev_c         = splitted[2 * layer_idx - 1]
    local prev_h         = splitted[2 * layer_idx]
    local next_c, next_h = lstm(i[layer_idx - 1], prev_c, prev_h,rnn_size)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    i[layer_idx] = next_h
  end
  local h2y              = nn.Linear(rnn_size, output_size)
  local pred             = nn.LogSoftMax()(h2y(i[layers]))
  local module           = nn.gModule({x, prev_s},
                                      {pred, nn.Identity()(next_s)})
  print ('useCuda ' ..useCuda.. '\n')
  
  if useCuda > 0 then
    module:cuda()
  end
  return module
end

function setup(seq_length, batch_size,layers, wordEmbedding_size, rnn_size,input_size, output_size,useCuda, init_weight)
  print("Creating a RNN LSTM network.")
  local core_network = create_network(layers, wordEmbedding_size, rnn_size,input_size, output_size, useCuda)
  local paramx, paramdx = core_network:getParameters()
  paramx:uniform(-init_weight,init_weight)
  local model = {}
  model.cost = nn.ClassNLLCriterion():cuda()
  
  model.paramx = paramx
  model.paramdx = paramdx
  model.s = {}
  model.ds = {}
  model.start_s = {}

  for j = 0, seq_length do
    model.s[j] = {}
    for d = 1, 2 * layers do
      if useCuda > 0 then
        model.s[j][d] = torch.zeros(batch_size, rnn_size):cuda()
      else
        model.s[j][d] = torch.zeros(batch_size, rnn_size)
      end
    end
  end
	
  for d = 1, 2 * layers do
    model.start_s[d] = torch.zeros(batch_size, rnn_size)
    model.ds[d] = torch.zeros(batch_size, rnn_size)
    if useCuda > 0 then
    	model.start_s[d] = torch.zeros(batch_size, rnn_size):cuda()
    	model.ds[d] = torch.zeros(batch_size, rnn_size):cuda()
    else
    	model.start_s[d] = torch.zeros(batch_size, rnn_size)
    	model.ds[d] = torch.zeros(batch_size, rnn_size)
    end
  end
  model.core_network = core_network
  
  --model.rnns = cloneManyTimes(core_network, seq_length)
  model.rnn_train = cloneManyTimes(core_network, 1)[1]
  
  model.norm_dw = 0
  reset_ds(model)
  return model
end

function reset_state(model)
  if model ~= nil and model.start_s ~= nil then
    for d = 1, #model.start_s do
      model.start_s[d]:zero()
    end
  end
end

function reset_ds(model)
  for d = 1, #model.ds do
    model.ds[d]:zero()
  end
end

function fp_bp(input, target, model, max_grad_norm, seq_length)
  --fp
  local tot_cost = 0
  copy_table(model.s[0], model.start_s)
  for i = 1, seq_length do
    local pred, state  = unpack(model.rnn_train:forward({input[i],model.s[i - 1]}))
    copy_table(model.s[i],state)
  end 
  copy_table(model.start_s, model.s[seq_length])
	
  --bp
  model.paramdx:zero()
  reset_ds(model)
  for i = seq_length, 1, -1 do
	local pred, state  = unpack(model.rnn_train:forward({input[i],model.s[i - 1]}))

    local cost = model.cost:forward(pred,target[i])
    tot_cost = tot_cost + cost
    local cost_grad = model.cost:backward(pred,target[i])

    local tmp = model.rnn_train:backward({input, model.s[i - 1]}, {cost_grad, model.ds})[2]
    copy_table(model.ds, tmp)
  end
  
  local norm_dw = model.paramdx:norm()

  if norm_dw > max_grad_norm then
	shrink_factor = max_grad_norm / norm_dw
    model.paramdx:mul(shrink_factor)
  end
 
  return tot_cost/seq_length
end


function main()
    local cmd = torch.CmdLine()
    
    cmd:option('-batch_size', 100, 'batch_size')
    cmd:option('-n_epochs', 2, 'n_epochs')
    cmd:option('-seq_length', 20, 'seq_length')
    cmd:option('-inputs_size', 654, 'dimension of input feature')
    cmd:option('-output_size', 654, 'output class')
    
    cmd:option('-wordEmbedding_size', 50, 'wordEmbedding_size')
    cmd:option('-layers', 2, 'layers')
    cmd:option('-rnn_size', 100, 'rnn_size')
    cmd:option('-init_weight', 0.08, 'init_weight')
    
    cmd:option('-learningRate', 0.01, 'learningRate')
    cmd:option('-learningRateReduce', 0.95, 'learningRateReduce')
    cmd:option('-min_gain_ratio', 0.2, 'min_gain_ratio')
    cmd:option('-min_learningRate', 0.00001, 'min_learningRate')
    
    cmd:option('-max_grad_norm', 5, 'max_grad_norm')
    cmd:option('-useCuda', 1, 'useCuda')
    
    cmd:option('-trainDataDir', '../RNNLM_Penn/trainData4RNN.npy', 'trainDataDir')
    cmd:option('-validDataDir', '../RNNLM_Penn/validData4RNN.npy', 'trainDataDir')
    cmd:option('-testDataDir', '../RNNLM_Penn/testData4RNN.npy', 'trainDataDir')
  
    cmd:text()
    local opt = cmd:parse(arg)
    torch.manualSeed(777)
    local batch_size=opt.batch_size
    local n_epochs=opt.n_epochs
    local seq_length=opt.seq_length
    local inputs_size = opt.inputs_size
    local output_size = opt.output_size
    local wordEmbedding_size = opt.wordEmbedding_size
    local layers = opt.layers
    local rnn_size = opt.rnn_size
    local init_weight = opt.init_weight
    
    local learningRate = opt.learningRate
    local learningRateReduce = opt.learningRateReduce
    local min_gain_ratio = opt.min_gain_ratio
    local min_learningRate = opt.min_learningRate
    local max_grad_norm = opt.max_grad_norm
    local useCuda = opt.useCuda

    local trainDataDir = opt.trainDataDir 
    local validDataDir = opt.validDataDir
    local testDataDir = opt.testDataDir
  
	
    showGPUMemory("Before setup")
    local model = setup(seq_length, batch_size,layers, wordEmbedding_size,rnn_size,inputs_size, output_size,useCuda,init_weight)
    showGPUMemory("After setup")
  
    --reset_state(model)
  
    local step = 0
    local epoch = 0
    local start_time = torch.tic()
    print("Starting training." ..'\n')
    epoch = 0
    train_done = false
    pre_xent4Train = 0.01
	
    local time = sys.clock()
    
    local trainArray = npy4th.loadnpy(trainDataDir)
    local n_train_batches = math.floor(trainArray:size(1)/(batch_size*seq_length))-1
    trainArray:resize(batch_size, trainArray:size(1)/batch_size)
	
    local validArray = npy4th.loadnpy(validDataDir)
    local n_valid_batches = math.floor(validArray:size(1)/(batch_size*seq_length))-1
    validArray:resize(batch_size, validArray:size(1)/batch_size)
	
    --training starts
    while(epoch < n_epochs) and (not train_done) do
        epoch=epoch+1
        for t = 1,n_train_batches do
                if train_done == true then
                    break
                end
                
                local input  = trainArray:narrow(2, 1+(t-1)*seq_length, seq_length)
                local target = trainArray:narrow(2, 2+(t-1)*seq_length, seq_length)
                input = input:transpose(1,2)
                target = target:transpose(1,2)
                
                function eval_training(paramx_)
                	local cost = fp_bp(input,target,model,max_grad_norm, seq_length)
                	return cost, model.paramdx
                end
                local _, cost = optim.sgd(eval_training, model.paramx, {learningRate=learningRate}, {})
                if t==1 then
                    showGPUMemory("first mini batch ")
                end  
        end

	print ('Learning rate ' ..learningRate.. ' at epoch '.. epoch ..'\n')
	--check error rate on training set after each epoch
	showGPUMemory("before test ")
	local xent4Train, perplexity4Train = valid_test(model, trainArray, n_train_batches, seq_length, batch_size)
	showGPUMemory("after test ")
	print ('xent on training set ' ..xent4Train.. ' at epoch '.. epoch ..'\n')
	print ('perplexity on training set ' ..perplexity4Train.. ' at epoch '.. epoch ..'\n')
                
        --check error rate on validation set after each epoch
        local xent4Valid, perplexity4Valid = valid_test(model, validArray, n_valid_batches, seq_length, batch_size)
	print ('xent on validation set ' ..xent4Valid.. ' at epoch '.. epoch ..'\n')
	print ('perplexity on validation set ' ..perplexity4Valid.. ' at epoch '.. epoch ..'\n')
        
        
    	--adjust learning rate
	if (pre_xent4Train - xent4Train)/pre_xent4Train < min_gain_ratio and epoch ~= 1 then
		learningRate=learningRate*learningRateReduce
	end
	if learningRate < min_learningRate then
        	train_done = true
        end
	pre_xent4Train = xent4Train
    end

    time = sys.clock() - time
    print ('Time used in training ' ..time/60 .. ' mins. Training is over. \n')
end

function validation(validFileTable, fileInTotal4Valid, model, isShuffleData, seq_length, max_day, ninputs,batch_size)
	local errorRate4Valid = 0.0
	local xent4Valid = 0.0
	for i = 1, fileInTotal4Valid do
		local input, target = loadData(validFileTable, i, isShuffleData, seq_length, max_day, ninputs)
		local errorRate, xent = valid_test(model, input, target, batch_size)

		errorRate4Valid = errorRate4Valid + errorRate
		xent4Valid = xent4Valid + xent
	end
	
	errorRate4Valid = errorRate4Valid/fileInTotal4Valid
	xent4Valid = xent4Valid/fileInTotal4Valid
	return errorRate4Valid, xent4Valid
end

function valid_test(model, dataArray, n_batches, seq_length, batch_size)
    reset_state(model)
    local corr_tot = 0
    local xent_tot = 0
    
    for t = 1, n_batches do
		local input  = dataArray:narrow(2, 1+(t-1)*seq_length, seq_length)
		local target = dataArray:narrow(2, 2+(t-1)*seq_length, seq_length)
		
		input = input:transpose(1,2)
		target = target:transpose(1,2)
        
        copy_table(model.s[0], model.start_s)
        
        for i = 1, seq_length do
        	local pred, state  = unpack(model.rnn_train:forward({input[i],model.s[i - 1]}))
        	local xent = model.cost:forward(pred,target[i])
			copy_table(model.s[i],state)
			
        	local predD1, predD2 = pred:max(2)
        	local target_cuda = torch.CudaLongTensor(target[i]:size())
        	local corr = torch.sum(torch.ne(predD2, target_cuda:copy(target[i])))
        
        	corr_tot = corr_tot + corr
        	xent_tot = xent_tot + xent
        end
        copy_table(model.start_s, model.s[seq_length])
    end
    
    --corr_tot*100.0/n_train_batches/batch_size/seq_length
    xent = xent_tot/n_batches/seq_length
    perplexity = math.exp(xent)
    return xent, perplexity
    
end

main()
