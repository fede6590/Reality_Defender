from tensorflow import keras

def create_data_aug_layer(data_aug_layer):
    """
    Use this function to parse the data augmentation methods for the
    experiment and create the corresponding layers.

    It will be mandatory to support at least the following three data
    augmentation methods (you can add more if you want):
        - `random_flip`: keras.layers.RandomFlip()
        - `random_rotation`: keras.layers.RandomRotation()
        - `random_zoom`: keras.layers.RandomZoom()
        - `random_contrast`: keras.layers.RandomContrast(),

    See https://tensorflow.org/tutorials/images/data_augmentation.

    Parameters
    ----------
    data_aug_layer : dict
        Data augmentation settings coming from the experiment YAML config
        file.

    Returns
    -------
    data_augmentation : keras.Sequential
        Sequential model having the data augmentation layers inside.
    """
    # Parse config and create layers
    # Append the data augmentation layers on this list

    data_augmentation = []

    # Supported data augmentations
    DATA_AUGMENTATIONS = {
    "random_flip": keras.layers.RandomFlip,
    "random_rotation": keras.layers.RandomRotation,
    "random_zoom": keras.layers.RandomZoom,
    "random_contrast": keras.layers.RandomContrast,
    }

    for type, params in data_aug_layer.items():
        data_augmentation.append(DATA_AUGMENTATIONS[type](**params))

    # Return a keras.Sequential model having the the new layers created
    data_augmentation = keras.Sequential(data_augmentation)

    return data_augmentation