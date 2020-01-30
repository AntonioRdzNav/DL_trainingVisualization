
#=====================================================================[ legal ]

"""
# Copyright (c) 2016-12-09, Laurenz Wiskott, www.ini.rub.de/PEOPLE/wiskott/
# License: BSD 2-Clause, https://opensource.org/licenses/BSD-2-Clause, see the end
"""

#====================================================================[ import ]

import sys                                    # sys utilities (e.g. exceptions)

import numpy as np                            # numpy for data handling
np.random.seed(42)                            # don't forget to seed the RNG!

from scipy.spatial.distance import euclidean  # euclidean distance function

#======================================================================[ util ]

"""
EXERCISE #01: Warmup

(a) Write a small utility function to check whether a given matrix is symmetric
    and whether its rows and columns add up to zero. Do not use any for() loops 
    and try to use numpy functionality wherever appropriate.

(b) Document your function! Throughout these exercises you will be asked to do
    this; take it as a chance to get into the habit of writing clean code! Here
    is an example of good documentation style:

        http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
"""

def check_laplacian( L ):
	"""
	Utility function to confirm basic properties of the laplacian matrix 
	(symmetry and rows/cols adding up to zero).

	Args:
		L (numpy.ndarray): (Laplacian) matrix to be checked.

	Returns:
		bool: Returns True if checks are successful; otherwise throws an
		      AssertionError.
	"""
	# 01: Symmetry
	check = all( [ all(L[i]==L[:,i]) for i in range(len(L)) ] )
	assert check is True, 'Laplacian matrix is not symmetric!'
	# 02: Rows / columns add up to zero.
	check = all( [ np.isclose( np.sum(L[i]), 0.0 ) for i in range(len(L)) ] )
	assert check is True, 'Rows of laplacian matrix do not sum to zero!'
	return True

#======================================================[ laplace matrix tools ]

"""
EXERCISE #02: Distance matrix

(a) Write a function that takes data set X and returns a matrix D where element
    D[i,j] holds the distance between X[i] and X[j].

(b) Add an additional parameter dst_func to your function which provides the 
    user the option to specify a custom function to use when computing the 
    distance between two elements. The value for dst_func should default to the
    euclidean distance.

(c) Since this computation can quickly become expensive, let's make use of the
    fact that D is symmetric. Adapt your function so that you only compute the
    distances in the lower (or upper) triangle of D and then mirror it across
    the diagonal.

(d) Document your function.
"""

def distance_matrix( x, dst_func=euclidean ):
	"""
	Computes a matrix where element D[i,j] holds the distance between data points
	x[i] and x[j] computed as dst_func( x[i], x[j] ).

	Args:
		x (numpy.ndarray): Input data; row x[i] holds samples i.

		dst_func (function): Function that returns a distance measure between
		                     two given data points. Can be used to compute a
		                     distance matrix based on a custom metric.

	Returns:
		numpy.ndarray: Distance matrix for data x.
	"""
	print( 'Computing distances between data points..' )
	D = np.zeros( [len(x), len(x)] )
	for i in range( len(x) ):
		D[i,:i] = [ dst_func(x[i], x[j])**2 for j in range(i) ]  # lower triangle values only (w/o diagonal)
	for i in range( len(x) ): D[i] = D[:,i]                      # mirror lower triangle to upper (d is symmetric)
	return D

"""
EXERCISE #03: Laplacian matrix

Write a function that takes data set X and computes a graph laplacian matrix
for it. The graph can be constructed by looking for the k nearest neighbours of
each data point, by using an epsilon-neighbourhoos approach, or simply be fully
connected. The weights of the graph can either be binary or exponentially decay
over the distance between two data points.

NOTE: Below is an example only; feel free to subdivide into utility functions
as desired and/or follow your own path to the solution as desired. In the end
though, a function with the provided signature and functionality should be
available (additional function parameters might be necessary).

! For all exercises make sure to not use any nested for loops and use fast 
! numpy routines wherever possible.

Proceed as follows (or whichever way you like):

(a) For the following exercises we want to work in place of the distance matrix
    D of our data set X. This is because D can become quite large, and we do 
    not want to have a working copy of it floating around.
    Begin by identifying the k nearest neighbours of each data point using D.
    Note that the neighbourhood relation is not symmetric -- why not? -- but
    we still need to end up with a simple graph, i.e., a symmetric weight 
    matrix.
    HINT: Take a look at numpy.partition(), it might come in handy.

(b) Once the k-nearest neighbours for each data point are identified, assign
    either binary or exponentially decaying weights to them and mask all other,
    non-used edges. (If X[i] and X[j] are not within each others neighbourhood
    D[i,j] = D[j,i] = 0.)

(c) Using our distance matrix turned weight matrix compute the graph laplacian 
    matrix L. Use your utility function to make sure that L is symmetric and
    rows/columns add up to zero.

(d) Implement the e-neighbourhood alternative to the k-nearest neighbourhood
    method of graph construction. To do so, similarly to (b), for each data 
    point identify all elements closer than a given distance e. Assign weights
    and null out all edges outside of each point's e-neighbourhood.

(e) In contrast to the k-nearest neighbourhood relation, the e-neighbourhood
    relation is symmetric. Apply the same trick as above and only identify the
    e-neighbourhoods for the lower/higher triangle elements of D, null out the
    unused edges, and mirror them to the other side.

(f) Make sure that it is easily possible to use an alternatice distance metric
    when computing the laplacian matrix.

(g) Make sure that it is easily possible to weigh edges differently once they
    have been established.

(h) Document your function.
"""

def laplacian( x, d=None, connections='k_nearest_neighbours', weights='binary', k=None ):
	"""
	Computes the graph laplacian matrix for data x.

	Args:
		x (numpy.ndarray): Input data. Row x[i] holds sample i.

		d (numpy.ndarray): Optional distance matrix. If not provided, default 
		                   (euclidean) distances will be computed and used. Can
		                   be used to base the graph on a custom distance 
		                   measure. [Default: None]

		connections (str): Method of connecting individual data points to form
		                   a graph. Available options are: 
		                   'k_nearest_neighbours','e_neighbourhood', or 'full'.
		                   [Default: 'k_nearest_neighbours']

		weights (str):     Method of weighing the edges of the graph. Available
						   options are: 'binary', 'exp_decay'. [Default: 'binary']

        k (float/int):     Generic parameter; use depends on the chose method.
                           Determines the k for k-nearest neighbours or the
                           epsilon for epsilon neighbourhood connectivity.
	"""

	# compute distance information if necessary
	if d is None and connections is not 'full':
		d = distance_matrix( x )

	# weight matrix for k nearest neighbours graph
	if connections is 'k_nearest_neighbours':

		print( 'Finding', k, 'nearest neighbours..' )
		for i in range( len(x) ):
			mask = d[i] <= np.partition( d[i], k )[k]
			if   weights is 'binary':    d[i][mask] = 1.0
			elif weights is 'exp_decay': d[i][mask] = np.exp( -d[i][mask]/k )
			else: raise ValueError( 'Unknown weight assignment: \''+str(weights)+'\'.' )
			d[i][np.invert(mask)] = 0.0
		
		print ( 'Fixing asymmetries in neighbourhood relations..' )
		for i in range( len(x) ):
			asym = ( d[i] != d[:,i] )
			d[i][asym] = d[:,i][asym] = np.abs( d[i][asym]-d[:,i][asym] )
		
	# weight matrix for e-neighbourhood graph
	elif connections is 'e_neighbourhood':
		
		print( 'Finding epsilon-neighbourhoods (e='+str(k)+')..' )
		for i in range( len(x) ):                # compute lower triangle values only
			mask = d[i,:i] < k
			if   weights is 'binary':    d[i,:i][mask] = 1.0
			elif weights is 'exp_decay': d[i,:i][mask] = np.exp( -d[i,:i][mask]/k )
			else: raise ValueError( 'Unknown weight assignment: \''+str(weights)+'\'.' )
			d[i,:i][np.invert(mask)] = 0.0
		for i in range( len(x) ): d[i] = d[:,i]  # copy lower triangle values to upper triangle

	# weight matrix for fully connected graph
	elif connections is 'full':
		if weights is 'binary':
			raise Warning( 'Using a fully connected graph and binary weights loses all information in the data!' )
		print( 'Computing weights for fully connected graph..' )
		d = np.exp( -d/k )

	# invalid choice for weight matrix
	else:
		raise ValueError('Unknown method for graph construction chosen.\n            ' + \
			             'Valid choices are: \'k_nearest_neighbours\', \'e_neighbourhood\', and \'full\'.')

	# compute the actual laplacian
	np.fill_diagonal( d, 0 )  # bulldoze the diagonal just in case anything snuck in there
	d = -d                    # construct the laplacian matrix
	d[ np.diag_indices(len(x)) ] = [ -np.sum(row) for row in d ]
	check_laplacian( d )      # sanity checks for the laplacian (symmetriy and zero-sums)
	return d                  # all done
#==============================================================================

# BSD 2-Clause License

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

#==============================================================================

