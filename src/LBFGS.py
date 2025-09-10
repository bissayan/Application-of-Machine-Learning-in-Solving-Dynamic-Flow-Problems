class L_BFGS_B:
    """
    Optimizer that implements the L-BFGS-B algorithm for training the PINN.
    
    This class handles the optimization process using scipy's L-BFGS-B implementation,
    tracks training metrics, and provides visualization of the convergence.
    """
    
    def __init__(self, model, x_train, y_train, factr=10, pgtol=1e-10, m=50, maxls=50, maxiter=5000):
        """
        Initialize the L-BFGS-B optimizer.
        
        Args:
            model: keras model to optimize
            x_train: training inputs
            y_train: training targets
            factr: convergence condition parameter
            pgtol: gradient convergence tolerance
            m: number of variable metric corrections
            maxls: maximum line search steps
            maxiter: maximum optimization iterations
        """
        self.model = model
        self.x_train = [tf.constant(x, dtype=tf.float32) for x in x_train]
        self.y_train = [tf.constant(y, dtype=tf.float32) for y in y_train]
        self.factr = factr
        self.pgtol = pgtol
        self.m = m
        self.maxls = maxls
        self.maxiter = maxiter
        self.metrics = ['loss']
        self.progbar = tf.keras.callbacks.ProgbarLogger()
        self.progbar.set_params({'verbose': 1, 'epochs': 1, 'steps': self.maxiter, 'metrics': self.metrics})
        self.loss_history = []
        self.epoch_history = []
        self.equation_loss_history = []
        self.continuity_loss_history = []
        self.boundary_loss_history = []

    def set_weights(self, flat_weights):
        """
        Set model weights from a flat numpy array.
        
        Args:
            flat_weights: 1D numpy array containing all model weights
        """
        shapes = [w.shape for w in self.model.get_weights()]
        split_ids = np.cumsum([np.prod(shape) for shape in [0] + shapes])
        weights = [flat_weights[from_id:to_id].reshape(shape) 
                  for from_id, to_id, shape in zip(split_ids[:-1], split_ids[1:], shapes)]
        self.model.set_weights(weights)

    @tf.function
    def tf_evaluate(self, x, y):
        """
        Compute loss and gradients for the current model weights.
        
        Args:
            x: input data
            y: target data
            
        Returns:
            total_loss: combined loss value
            grads: gradients of trainable variables
            loss_u_v: momentum equation loss
            loss_psi: continuity equation loss
            loss_boundary: boundary condition loss
        """
        alpha = 1  # Weight for u and v equations
        beta = 1   # Weight for continuity equation (psi)
        gamma = 0.1  # Weight for boundary data

        with tf.GradientTape() as g:
            y_pred = self.model(x)
            loss_u_v = alpha * tf.reduce_mean(tf.square(y_pred[0] - y[0]))
            loss_psi = beta * tf.reduce_mean(tf.square(y_pred[1] - y[1]))
            loss_boundary = gamma * tf.reduce_mean(tf.square(y_pred[2] - y[2]))
            total_loss = loss_u_v + loss_psi + loss_boundary

        grads = g.gradient(total_loss, self.model.trainable_variables)
        return total_loss, grads, loss_u_v, loss_psi, loss_boundary

    def evaluate(self, weights):
        """
        Evaluate function for scipy.optimize.
        
        Args:
            weights: current weight values as flat numpy array
            
        Returns:
            total_loss: scalar loss value
            grads: flat numpy array of gradients
        """
        self.set_weights(weights)
        total_loss, grads, loss_u_v, loss_psi, loss_boundary = self.tf_evaluate(self.x_train, self.y_train)
        total_loss = total_loss.numpy().astype('float64')
        grads = np.concatenate([g.numpy().flatten() for g in grads]).astype('float64')

        # Store loss values for tracking
        self.loss_history.append(total_loss)
        self.equation_loss_history.append(loss_u_v.numpy())
        self.continuity_loss_history.append(loss_psi.numpy())
        self.boundary_loss_history.append(loss_boundary.numpy())
        self.epoch_history.append(len(self.loss_history))

        return total_loss, grads

    def callback(self, weights):
        """Callback function to print progress during optimization."""
        self.progbar.on_batch_begin(0)
        total_loss = self.loss_history[-1]
        loss_u_v = self.equation_loss_history[-1]
        loss_psi = self.continuity_loss_history[-1]
        loss_boundary = self.boundary_loss_history[-1]
        print(f"Epoch: {len(self.loss_history)}, Total Loss: {total_loss}, "
              f"Loss (u, v): {loss_u_v}, Loss (psi): {loss_psi}, Loss (Boundary): {loss_boundary}")
        self.progbar.on_batch_end(0, logs=dict(zip(self.metrics, [total_loss])))

    def fit(self):
        """Run the L-BFGS-B optimization."""
        initial_weights = np.concatenate([w.flatten() for w in self.model.get_weights()])
        print('Optimizer: L-BFGS-B (maxiter={})'.format(self.maxiter))
        self.progbar.on_train_begin()
        self.progbar.on_epoch_begin(1)
        scipy.optimize.fmin_l_bfgs_b(
            func=self.evaluate, 
            x0=initial_weights,
            factr=self.factr, 
            pgtol=self.pgtol, 
            m=self.m,
            maxls=self.maxls, 
            maxiter=self.maxiter, 
            callback=self.callback
        )
        self.progbar.on_epoch_end(1)
        self.progbar.on_train_end()
        self.plot_convergence()

    def plot_convergence(self):
        """Plot the convergence history of the training process."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.epoch_history, self.loss_history, label='Total Loss', linestyle='-', linewidth=2)
        plt.plot(self.epoch_history, self.equation_loss_history, label='Loss (u, v)', linestyle='--')
        plt.plot(self.epoch_history, self.continuity_loss_history, label='Loss (psi)', linestyle='--')
        plt.plot(self.epoch_history, self.boundary_loss_history, label='Loss (Boundary)', linestyle='--')

        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.1e}'))

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Convergence Plot')
        plt.legend()
        plt.grid(True)
        plt.show()