"""Forecasting method interface and registry."""#-#
#-#
from sim_integration.methods.base import ForecastMethod#-#
from sim_integration.methods.const_vel import ConstantVelocityMethod#-#
#-#
__all__ = ["ForecastMethod", "ConstantVelocityMethod"]#-#
