class ConcatLayer(tf.keras.layers.Layer):
    """
    Simple custom layer to concatenate tensors along the last axis.
    """
    
    def call(self, inputs):
        """
        Concatenate input tensors.
        
        Args:
            inputs: list of tensors to concatenate
            
        Returns:
            Concatenated tensor
        """
        return tf.concat(inputs, axis=-1)