
import numpy as np
import jax.numpy as jnp
from jax import jit

class PchipInterpolatorJax:
    def __init__(self, x: jnp.ndarray, y: jnp.ndarray):
        if len(x.shape) != 1 or len(y.shape) != 1:
            raise ValueError("x and y arrays must be 1-dimensional")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y arrays must have the same length")
        if x.shape[0] < 2:
            raise ValueError("arrays must contain at least 2 points")
        
        self.x = x
        self.y = y
        self.h = jnp.diff(x)
        self.dy = jnp.diff(y)
        self.slopes = self.dy / self.h
        
        # Calculate derivatives
        self.d = self._compute_derivatives()
    
    def _compute_derivatives(self) -> jnp.ndarray:
        """
        Compute the derivatives at each point using the PCHIP algorithm
        """
        n = len(self.x)
        d = jnp.zeros_like(self.x)
        
        # Interior points
        w1 = 2 * self.h[1:] + self.h[:-1]
        w2 = self.h[1:] + 2 * self.h[:-1]
        
        # Weighted harmonic mean of slopes
        s = jnp.where(
            self.slopes[:-1] * self.slopes[1:] > 0,
            1 / ((w1 / (3 * self.h[1:] * self.slopes[:-1]) + 
                  w2 / (3 * self.h[:-1] * self.slopes[1:]))),
            0.0
        )
        d = d.at[1:-1].set(s)
        
        # End points
        # Left endpoint
        if n > 2:
            d = d.at[0].set(
                jnp.where(
                    self.slopes[0] * self.slopes[1] > 0,
                    ((2 * self.h[0] + self.h[1]) * self.slopes[0] - self.h[0] * self.slopes[1]) / (self.h[0] + self.h[1]),
                    3 * self.slopes[0] * (1 - self.h[0] / (self.h[0] + self.h[1]))
                )
            )
            
            # Right endpoint
            d = d.at[-1].set(
                jnp.where(
                    self.slopes[-1] * self.slopes[-2] > 0,
                    ((2 * self.h[-1] + self.h[-2]) * self.slopes[-1] - self.h[-1] * self.slopes[-2]) / (self.h[-1] + self.h[-2]),
                    3 * self.slopes[-1] * (1 - self.h[-1] / (self.h[-1] + self.h[-2]))
                )
            )
        else:
            # For n == 2, use simple derivatives
            d = d.at[0].set(self.slopes[0])
            d = d.at[-1].set(self.slopes[0])
        
        return d
    
    def __call__(self, xi: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the interpolant at points xi
        """
        indices = jnp.searchsorted(self.x[1:], xi)
        
        x_left = jnp.take(self.x, indices)
        x_right = jnp.take(self.x, indices + 1)
        h = x_right - x_left
        t = (xi - x_left) / h
        
        y_left = jnp.take(self.y, indices)
        y_right = jnp.take(self.y, indices + 1)
        d_left = jnp.take(self.d, indices)
        d_right = jnp.take(self.d, indices + 1)
        
        # Hermite basis functions
        t2 = t * t
        t3 = t2 * t
        
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        
        return h00 * y_left + h10 * h * d_left + h01 * y_right + h11 * h * d_right

@jit
def interpolate(x: jnp.ndarray, y: jnp.ndarray, xi: jnp.ndarray) -> jnp.ndarray:
    """
    Convenience function for PCHIP interpolation
    """
    interpolator = PchipInterpolatorJax(x, y)
    return interpolator(xi)
