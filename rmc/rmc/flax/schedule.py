# -*- coding: utf-8 -*-

"""Utilities for defining time schedules for flows."""

import abc
import jax
import jax.numpy as jnp

class BaseSchedule(metaclass=abc.ABCMeta):
    """Base class for defining time schedules for flows."""
    
    def __init__(self):
        pass
        
        
    def __call__(self, t):
        """Evaluate schedule function.
        
        Args:
            t: Time to evaluate schedule function.
            
        Returns: Schedule function and derivative of schedule function evaluated at t.
        """
        outtau = self.tau(t)
        douttau = self.dtau(t)
        
        return outtau, douttau
        
        
    @abc.abstractmethod
    def tau(self, t):
        """Definition of schedule function.
        
        Args:
            t: Time to evaluate schedule function.
            
        Returns: 
            Schedule function evaluated at t.        
        """
        pass
        
        
    def dtau(self, t):
        """Definition of derivative of schedule function.
        
        Args:
            t: Time to evaluate schedule function.
            
        Returns: 
            Derivative of schedule function evaluated at t.
        """
        return jax.grad(self.tau)(t)



class CosineSchedule(BaseSchedule):
    """Class for defining a cosine schedule."""
    
    def __init__(self):
        super().__init__()
        
        
    def tau(self, t):
        """Definition of schedule function.
        
        Args:
            t: Time to evaluate schedule function.
            
        Returns: 
            Schedule function evaluated at t.        
        """
        return 0.5 * (1.0 - jnp.cos(jnp.pi * t))
        
