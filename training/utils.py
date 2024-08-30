
import tensorflow.keras.backend as K

# Determine the dice coeff for each class

def dice_coefficient_each_class(y_true, y_pred, class_index, smooth=1e-6):
    y_true_f = K.flatten(y_true[..., class_index])
    y_pred_f = K.flatten(y_pred[..., class_index])
    
    intersection = K.sum(y_true_f * y_pred_f)
    dice_numerator = 2. * intersection
    dice_denominator = K.sum(y_true_f) + K.sum(y_pred_f)
    
    dice_score = (dice_numerator + smooth) / (dice_denominator + smooth)
    return dice_score

def dice_coefficient_necrotic(y_true, y_pred):
    return dice_coefficient_each_class(y_true, y_pred, class_index=1)

def dice_coefficient_edema(y_true, y_pred):
    return dice_coefficient_each_class(y_true, y_pred, class_index=2)

def dice_coefficient_enhancing(y_true, y_pred):
    return dice_coefficient_each_class(y_true, y_pred, class_index=3)