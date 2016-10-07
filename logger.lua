
lossLogger = optim.Logger(paths.concat(opt.save, 'loss.log'))
lossLogger:setNames{'% mean loss (train set)', '% mean loss (val set)'}
lossLogger:style{'+-', '+-'}
lossLogger.showPlot = false

reg_accuracyLogger = optim.Logger(paths.concat(opt.save, 'reg_accuracy.log'))
reg_accuracyLogger:setNames{'% regression accuracy (train set)', '% regression accuracy (val set)'}
reg_accuracyLogger:style{'+-', '+-'}
reg_accuracyLogger.showPlot = false

false_posLogger = optim.Logger(paths.concat(opt.save, 'false_pos.log'))
false_posLogger:setNames{'% false positives (train set)', '% false positives (val set)'}
false_posLogger:style{'+-', '+-'}
false_posLogger.showPlot = false

false_negLogger = optim.Logger(paths.concat(opt.save, 'false_neg.log'))
false_negLogger:setNames{'% false negatives (train set)', '% false negatives (val set)'}
false_negLogger:style{'+-', '+-'}
false_negLogger.showPlot = false

corr_classLogger = optim.Logger(paths.concat(opt.save, 'corr_class.log'))
corr_classLogger:setNames{'% false negatives (train set)', '% false negatives (val set)'}
corr_classLogger:style{'+-', '+-'}
corr_classLogger.showPlot = false

reg_lossLogger = optim.Logger(paths.concat(opt.save, 'reg_loss.log'))
reg_lossLogger:setNames{'% reg loss (train set)', '% reg loss(val set)'}
reg_lossLogger:style{'+-', '+-'}
reg_lossLogger.showPlot = false

conf_lossLogger = optim.Logger(paths.concat(opt.save, 'conf_loss.log'))
conf_lossLogger:setNames{'% conf loss (train set)', '% conf loss (val set)'}
conf_lossLogger:style{'+-', '+-'}
conf_lossLogger.showPlot = false

neg_lossLogger = optim.Logger(paths.concat(opt.save, 'neg_loss.log'))
neg_lossLogger:setNames{'% neg loss (train set)', '% neg loss (val set)'}
neg_lossLogger:style{'+-', '+-'}
neg_lossLogger.showPlot = false

class_lossLogger = optim.Logger(paths.concat(opt.save, 'class_loss.log'))
class_lossLogger:setNames{'% class loss (train set)', '% class loss (val set)'}
class_lossLogger:style{'+-', '+-'}
class_lossLogger.showPlot = false

train_loss = 0
train_reg_accuracy = 0
train_false_pos = 0
train_false_neg = 0
train_corr_class = 0

val_reg_accuracy = 0
val_false_pos = 0
val_false_neg = 0
val_corr_class = 0
val_loss = 0

e_class = 0
e_reg = 0
e_conf = 0
e_neg = 0

val_class = 0
val_reg = 0
val_conf = 0
val_neg = 0

learning_rate_shedule = {}

local loss_dir = paths.concat(opt.save, 'loss.png')      
local reg_accuracy_dir = paths.concat(opt.save, 'reg_accuracy.png')
local false_pos_dir = paths.concat(opt.save, 'false_pos.png') 
local false_neg_dir = paths.concat(opt.save, 'false_neg.png')
local corr_class_dir = paths.concat(opt.save, 'corr_class.png')
local reg_loss_dir = paths.concat(opt.save, 'reg_loss.png')
local conf_loss_dir = paths.concat(opt.save, 'conf_loss.png')
local neg_loss_dir = paths.concat(opt.save, 'neg_loss.png')
local class_loss_dir = paths.concat(opt.save, 'class_loss.png')

function logging()

    lossLogger:add{train_loss, val_loss}
    lossLogger:plot()
    reg_accuracyLogger:add{train_reg_accuracy, val_reg_accuracy}
    reg_accuracyLogger:plot()
    false_posLogger:add{train_false_pos, val_false_pos}
    false_posLogger:plot()
    false_negLogger:add{train_false_neg, val_false_neg}
    false_negLogger:plot()
    corr_classLogger:add{train_corr_class, val_corr_class}
    corr_classLogger:plot()
    reg_lossLogger:add{e_reg, val_reg}
    reg_lossLogger:plot()
    conf_lossLogger:add{e_conf, val_conf}
    conf_lossLogger:plot()
    neg_lossLogger:add{e_neg, val_neg}
    neg_lossLogger:plot()
    class_lossLogger:add{e_class, val_class}
    class_lossLogger:plot()
end


function writeReport()

  os.execute(('convert -density 200 %sloss.log.eps %sloss.png'):format(opt.save,opt.save))
  os.execute(('convert -density 200 %sreg_accuracy.log.eps %sreg_accuracy.png'):format(opt.save,opt.save))
  os.execute(('convert -density 200 %sfalse_pos.log.eps %sfalse_pos.png'):format(opt.save,opt.save))
  os.execute(('convert -density 200 %sfalse_neg.log.eps %sfalse_neg.png'):format(opt.save,opt.save))
  os.execute(('convert -density 200 %scorr_class.log.eps %scorr_class.png'):format(opt.save,opt.save))
  os.execute(('convert -density 200 %sreg_loss.log.eps %sreg_loss.png'):format(opt.save,opt.save))
  os.execute(('convert -density 200 %sconf_loss.log.eps %sconf_loss.png'):format(opt.save,opt.save))
  os.execute(('convert -density 200 %sneg_loss.log.eps %sneg_loss.png'):format(opt.save,opt.save))
  os.execute(('convert -density 200 %sclass_loss.log.eps %sclass_loss.png'):format(opt.save,opt.save))

  local file = io.open(opt.save..'report.html','w')
  file:write(([[
      <!DOCTYPE html>
      <html>
      <body>
      <h4>Log: %s</h4>
      <h4>Epoch: %s</h4>
      <h4> Loss: </h4>
      <img src=%s>
      <h4> Accuracy regression: </h4>
      <img src=%s>
      <h4> False positives: </h4>
      <img src=%s>
      <h4> False negatives</h4>
      <img src=%s>
      <h4> Correct Class</h4>
      <img src=%s>
      <h4> Reg Loss</h4>
      <img src=%s>
      <h4> Conf Loss</h4>
      <img src=%s>
      <h4> Neg Loss</h4>
      <img src=%s>
      <h4> Class Loss</h4>
      <img src=%s>
      <h4> Accuracies</h4>
      <table>
      ]]):format(opt.save,epoch, 'loss.png', 'reg_accuracy.png','false_pos.png', 'corr_class.png', 'reg_loss.png', 'conf_loss.png', 'neg_loss.png', 'neg_loss.png', 'class_loss.png'))      

  file:write('<tr> <td>'..'Regression_accuracy_val: '..'</td> <td>'.. val_reg_accuracy ..'</td> </tr> \n')
  file:write('<tr> <td>'..'Regression_accuracy_train: '..'</td> <td>'.. train_reg_accuracy ..'</td> </tr> \n')
  file:write('<tr> <td>'..'Val false positives:'..'</td> <td>'.. val_false_pos ..'</td> </tr> \n')
  file:write('<tr> <td>'..'Val false negatives:'..'</td> <td>'.. val_false_neg ..'</td> </tr> \n')
  file:write('<tr> <td>'..'Val correct labeled (only gt pos):'..'</td> <td>'.. val_corr_class ..'</td> </tr> \n')
  file:write('<tr> <td>'..'Regression loss:'..'</td> <td>'.. e_reg ..'</td> </tr> \n')
  

-----------------------------------------------------------------------------------------

  file:write([[</table>
  <h4> OptimState: </h4>
  <table>
  ]])

  for k,v in pairs(optimState) do
    if torch.type(v) == 'number' then
      file:write('<tr> <td>'..k..'</td> <td>'..v..'</td> </tr> \n')
    end
  end

-----------------------------------------------------------------------------------------

  file:write([[</table>
  <h4> Opts: </h4>
  <table>
  ]])

  for k,v in pairs(opt) do
    if torch.type(v) == 'number' or torch.type(v) == 'string' then
      file:write('<tr> <td>'..k..'</td> <td>'..v..'</td> </tr> \n')
    end
  end

-----------------------------------------------------------------------------------------
if opt.LR == 0.0 then
	  file:write([[</table>
	  <h4> Learning Rate Shedule: </h4>
	  <table>
	  ]])
    
    file:write('<tr> <td>'..'Begin epoch' ..'</td> <td>'..'End epoch' ..'</td><td>'..'learningRate' ..'</td><td>'..'WeightDecy' ..'</td> </tr> \n')
	
	  for k,v in pairs(learning_rate_shedule) do
      file:write(('<tr> <td> %d </td> <td> %d </td> <td> %.4f </td> <td> %.4f </td> </tr> \n'):format(v[1], v[2], v[3], v[4]))
	  end
    
end
-----------------------------------------------------------------------------------------

  file:write([[
    </table>
    <h4> Train Images </h4>
    gt image  - gt und prediction image </br>
    ]])

--input and output images
  for i = 1, 12 do
    input_dir = 'Images/trainGt' .. i .. '.png'      
    label_dir = 'Images/trainEx' .. i .. '.png'   

    file:write(([[
      <h5> ImagePair: %s </h5>    
      <img src=%s>
      <img src=%s>
      </br>
      ]]):format(i, input_dir, label_dir))     
  end

-----------------------------------------------------------------------------------------

  file:write([[</table>
    <h4> Model: </h4>
    <pre> ]])
  file:write(tostring(model))

  file:write'</pre></body></html>'
  file:close()

end
