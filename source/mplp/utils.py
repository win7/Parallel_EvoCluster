# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Copyright (c) 2017, Vincent Gauthier, Institut Mines Telecom/Telecom SudParis.
# All rights reserved.                                                          
#                                                                               
# Redistribution and use in source and binary forms, with or without            
# modification, are permitted provided that the following conditions are met:   
#                                                                               
# * Redistributions of source code must retain the above copyright notice, this 
#   list of conditions and the following disclaimer.                            
#                                                                               
# * Redistributions in binary form must reproduce the above copyright notice,   
#   this list of conditions and the following disclaimer in the documentation   
#   and/or other materials provided with the distribution.                      
#                                                                               
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 'AS IS'   
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE     
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE  
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL    
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR    
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER    
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.          
# ------------------------------------------------------------------------------

__author__ = "Vincent Gauthier <vgauthier@luxbulb.org>"
__license__ = "MIT"
__all__ = ['logbins']

def logbins(x):
    """Find the logarithmic bin size of a given dataset found the code in [1]

    Parameters
    ----------
        x: ndarray
            data to be bined  
    
    Return
    ------
        bins: ndarray 
            bins with logarithmic size

    References:
        [1] Jeff Alstott, Ed Bullmore, Dietmar Plenz. (2014). powerlaw: a Python package for analysis of heavy-tailed distributions. https://github.com/jeffalstott/powerlaw

    """
    from numpy import logspace, floor, unique
    from math import log10, ceil

    log_min_size = log10(x.min())
    log_max_size = log10(x.max())
    number_of_bins = ceil((log_max_size-log_min_size)*10)
    bins=unique(floor(logspace(log_min_size, log_max_size, num=number_of_bins)))
    return bins