require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'
require 'cudnn'
require 'cunn'

dofile('utils/util_calc_box_from_reg.lua')
dofile('utils/util_calc_grid_index.lua')
dofile('utils/util_generate_batch_roidbs.lua')
dofile('utils/util_boxoverlap.lua')
dofile('utils/util_flip_rois.lua')
dofile('utils/util_scale_rois.lua')
dofile('utils/util_restrict_rois_to_image_size.lua')
dofile('utils/util_img_from_mean.lua')
dofile('utils/util_box_size.lua')
dofile('utils/util_binaryToIdx.lua')
dofile('prepare_image_roidb.lua')

torch.setdefaulttensortype('torch.FloatTensor')

local opts = paths.dofile('opts.lua')

opt = opts.parse(arg)

nClasses = opt.nClasses

paths.dofile('util.lua')
paths.dofile('model.lua')
opt.imageSize = model.imageSize or opt.imageSize
opt.imageCrop = model.imageCrop or opt.imageCrop

print(opt)

cutorch.setDevice(opt.GPU) -- by default, use GPU 1
torch.manualSeed(opt.manualSeed)

print('==> load Data')
paths.dofile('data.lua')
print('Loaded data')
paths.dofile('donkey.lua')
paths.dofile('logger.lua')
paths.dofile('calc_correct_output.lua')

if opt.crossEntropy then
  paths.dofile('train_cec.lua')
else
  paths.dofile('train.lua')
end

--paths.dofile('validate.lua')
--paths.dofile('test_output.lua')
--paths.dofile('test.lua')

epoch = opt.epochNumber

print('start training')
for i=1,opt.nEpochs do
   train()
   if opt.do_validation then
     validate()
   end
   logging()
   writeReport()
   epoch = epoch + 1
end