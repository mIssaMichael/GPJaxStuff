from __future__ import annotations


# import equinox as eqx
# import distrax as dax

# Note: parameters should inherit from transformed (similar to Stan {parameters} and {transformed parameters} blocking functionality)
# Split into two files




import dataclasses
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

class Transform(ABC):
    """Base class for parameter transformations."""

    @abstractmethod
    def forward(self, x: ArrayLike) -> ArrayLike:
        """Transform from unconstrained to constrained space."""
        pass

    @abstractmethod
    def inverse(self, y: ArrayLike) -> ArrayLike:
        """Transform from constrained to unconstrained space."""
        pass

    @abstractmethod
    def forward_log_det_jacobian(self, x: ArrayLike) -> ArrayLike:
        """Log determinant of Jacobian of forward transform."""
        pass

class Identity(Transform):
    """Identity transform for unconstrained parameters."""

    def forward(self, x: ArrayLike) -> ArrayLike:
        return x

    def inverse(self, y: ArrayLike) -> ArrayLike:
        return y

    def forward_log_det_jacobian(self, x: ArrayLike) -> ArrayLike:
        return jnp.zeros_like(x)

class LowerBoundedTransform(Transform):
    """Transform for parameters with a lower bound."""

    def __init__(self, lower_bound: float):
        self.lower_bound = jnp.float32(lower_bound)

    def forward(self, x: ArrayLike) -> ArrayLike:
        return jnp.add(self.lower_bound, jax.nn.softplus(x))

    def inverse(self, y: ArrayLike) -> ArrayLike:
        shifted = jnp.subtract(y, self.lower_bound)
        return jnp.log(jnp.subtract(jnp.exp(shifted), 1.0))

    def forward_log_det_jacobian(self, x: ArrayLike) -> ArrayLike:
        return jnp.negative(jnp.log1p(jnp.exp(jnp.negative(x))))

class UpperBoundedTransform(Transform):
    """Transform for parameters with an upper bound."""

    def __init__(self, upper_bound: float):
        self.upper_bound = jnp.float32(upper_bound)

    def forward(self, x: ArrayLike) -> ArrayLike:
        return jnp.subtract(self.upper_bound, jax.nn.softplus(x))

    def inverse(self, y: ArrayLike) -> ArrayLike:
        shifted = jnp.subtract(self.upper_bound, y)
        return jnp.log(jnp.subtract(jnp.exp(shifted), 1.0))

    def forward_log_det_jacobian(self, x: ArrayLike) -> ArrayLike:
        return jnp.negative(jnp.log1p(jnp.exp(jnp.negative(x))))

class IntervalTransform(Transform):
    """Transform for parameters in a finite interval."""

    def __init__(self, lower_bound: float, upper_bound: float):
        if lower_bound >= upper_bound:
            raise ValueError(
                f"Lower bound ({lower_bound}) must be less than upper bound ({upper_bound})"
            )
        self.lower_bound = jnp.float32(lower_bound)
        self.upper_bound = jnp.float32(upper_bound)
        self.scale = jnp.subtract(upper_bound, lower_bound)

    def forward(self, x: ArrayLike) -> ArrayLike:
        sigmoid_x = jax.nn.sigmoid(x)
        scaled = jnp.multiply(self.scale, sigmoid_x)
        return jnp.add(self.lower_bound, scaled)

    def inverse(self, y: ArrayLike) -> ArrayLike:
        shifted = jnp.subtract(y, self.lower_bound)
        y_scaled = jnp.divide(shifted, self.scale)
        return jnp.log(jnp.divide(y_scaled, jnp.subtract(1.0, y_scaled)))

    def forward_log_det_jacobian(self, x: ArrayLike) -> ArrayLike:
        log_scale = jnp.log(self.scale)
        log_sigmoid = jax.nn.log_sigmoid(x)
        log_sigmoid_neg = jax.nn.log_sigmoid(jnp.negative(x))
        return jnp.add(log_scale, jnp.add(log_sigmoid, log_sigmoid_neg))

class AffineTransform(Transform):
    """Transform for affinely transformed parameters."""

    def __init__(self, scale: float = 1.0, offset: float = 0.0):
        self.scale = jnp.float32(scale)
        self.offset = jnp.float32(offset)

    def forward(self, x: ArrayLike) -> ArrayLike:
        return jnp.add(jnp.multiply(self.scale, x), self.offset)

    def inverse(self, y: ArrayLike) -> ArrayLike:
        shifted = jnp.subtract(y, self.offset)
        return jnp.divide(shifted, self.scale)

    def forward_log_det_jacobian(self, x: ArrayLike) -> ArrayLike:
        return jnp.full_like(x, jnp.log(jnp.abs(self.scale)))

class OrderedTransform(Transform):
    """Transform for ordered vectors."""

    def forward(self, x: ArrayLike) -> ArrayLike:
        return jnp.cumsum(jax.nn.softplus(x), axis=-1)

    def inverse(self, y: ArrayLike) -> ArrayLike:
        diffs = jnp.diff(y, axis=-1)
        return jnp.log(jnp.subtract(jnp.exp(diffs), 1.0))

    def forward_log_det_jacobian(self, x: ArrayLike) -> ArrayLike:
        return jnp.negative(jnp.sum(
            jnp.log1p(jnp.exp(jnp.negative(x))),
            axis=-1
        ))

class UnitSimplexTransform(Transform):
    """Transform for vectors on the unit simplex."""

    def forward(self, x: ArrayLike) -> ArrayLike:
        size = jnp.shape(x)[-1]
        z = jax.nn.sigmoid(x)

        # Get all elements except the last one
        z_head = jax.lax.dynamic_slice_in_dim(z, 0, size - 1, axis=-1)
        z_comp = jnp.subtract(1.0, z_head)
        z_cumprod = jnp.cumprod(z_comp, axis=-1)

        # Compute the stick-breaking proportions
        proportions = jnp.multiply(z_head, z_cumprod)
        last_prop = jax.lax.dynamic_slice_in_dim(z_cumprod, size - 2, 1, axis=-1)

        return jnp.concatenate([proportions, last_prop], axis=-1)
    def inverse(self, y: ArrayLike) -> ArrayLike:
        size = jnp.shape(y)[-1]
        y_head = jax.lax.dynamic_slice_in_dim(jnp.asarray(y), 0, size - 1, axis=-1)

        z_cumprod = jnp.cumprod(jnp.subtract(1.0, y_head), axis=-1)
        z = jnp.divide(y_head, z_cumprod)

        return jax.nn.log_sigmoid(z)

    def forward_log_det_jacobian(self, x: ArrayLike) -> ArrayLike:
        log_sigmoid = jax.nn.log_sigmoid(x)
        log_sigmoid_neg = jax.nn.log_sigmoid(jnp.negative(x))
        return jnp.sum(jnp.add(log_sigmoid, log_sigmoid_neg), axis=-1)

class StochasticMatrixTransform(Transform):
    """Transform for stochastic matrices (rows sum to 1)."""

    def __init__(self):
        self.simplex_transform = UnitSimplexTransform()

    def forward(self, x: ArrayLike) -> ArrayLike:
        return jax.vmap(self.simplex_transform.forward)(x)

    def inverse(self, y: ArrayLike) -> ArrayLike:
        return jax.vmap(self.simplex_transform.inverse)(y)

    def forward_log_det_jacobian(self, x: ArrayLike) -> ArrayLike:
        return jnp.sum(
            jax.vmap(self.simplex_transform.forward_log_det_jacobian)(x)
        )

class UnitVectorTransform(Transform):
    """Transform for unit vectors."""

    def forward(self, x: ArrayLike) -> ArrayLike:
        norm = jnp.sqrt(jnp.sum(jnp.square(x), axis=-1, keepdims=True))
        return jnp.divide(x, norm)

    def inverse(self, y: ArrayLike) -> ArrayLike:
        return y

    def forward_log_det_jacobian(self, x: ArrayLike) -> ArrayLike:
        n = jnp.shape(x)[-1]
        norm = jnp.sqrt(jnp.sum(jnp.square(x), axis=-1))
        return jnp.multiply(
            jnp.subtract(1, n),
            jnp.log(norm)
        )

@dataclasses.dataclass
class Parameter:
    """Parameter with transform."""
    value: ArrayLike
    transform: Transform = dataclasses.field(default_factory=Identity)

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if not isinstance(self.value, (jnp.ndarray, float, int)):
            raise TypeError(f"Expected array-like value, got {type(self.value)}")

    @property
    def constrained_value(self) -> ArrayLike:
        return self.transform.forward(self.value)

    @property
    def unconstrained_value(self) -> ArrayLike:
        return self.value

    def set_constrained(self, value: ArrayLike) -> Parameter:
        return type(self)(
            self.transform.inverse(value),
            transform=self.transform
        )
