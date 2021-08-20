cmake_minimum_required (VERSION 3.10.2)

##### PLUGIN NAME #####
set(pluginname gipOpenNN)


##### PLUGIN PATHS #####
set(PLUGIN_DIR ${PLUGINS_DIR}/${pluginname})
set(PREBUILT_DIR ${PLUGIN_DIR}/prebuilts)


##### PLUGIN SOURCES #####
list(APPEND PLUGIN_SRCS
			${PLUGIN_DIR}/src/gipOpenNN.cpp
			${PLUGIN_DIR}/opennn/adaptive_moment_estimation.cpp
			${PLUGIN_DIR}/opennn/bounding_layer.cpp
			${PLUGIN_DIR}/opennn/conjugate_gradient.cpp
			${PLUGIN_DIR}/opennn/convolutional_layer.cpp
			${PLUGIN_DIR}/opennn/correlations.cpp
			${PLUGIN_DIR}/opennn/cross_entropy_error.cpp
			${PLUGIN_DIR}/opennn/data_set.cpp
			${PLUGIN_DIR}/opennn/genetic_algorithm.cpp
			${PLUGIN_DIR}/opennn/gradient_descent.cpp
			${PLUGIN_DIR}/opennn/growing_inputs.cpp
			${PLUGIN_DIR}/opennn/growing_neurons.cpp
			${PLUGIN_DIR}/opennn/inputs_selection.cpp
			${PLUGIN_DIR}/opennn/layer.cpp
			${PLUGIN_DIR}/opennn/learning_rate_algorithm.cpp
			${PLUGIN_DIR}/opennn/levenberg_marquardt_algorithm.cpp
			${PLUGIN_DIR}/opennn/long_short_term_memory_layer.cpp
			${PLUGIN_DIR}/opennn/loss_index.cpp
			${PLUGIN_DIR}/opennn/mean_squared_error.cpp
			${PLUGIN_DIR}/opennn/minkowski_error.cpp
			${PLUGIN_DIR}/opennn/model_selection.cpp
			${PLUGIN_DIR}/opennn/neural_network.cpp
			${PLUGIN_DIR}/opennn/neurons_selection.cpp
			${PLUGIN_DIR}/opennn/normalized_squared_error.cpp
			${PLUGIN_DIR}/opennn/numerical_differentiation.cpp
			${PLUGIN_DIR}/opennn/opennn_strings.cpp
			${PLUGIN_DIR}/opennn/optimization_algorithm.cpp
			${PLUGIN_DIR}/opennn/perceptron_layer.cpp
			${PLUGIN_DIR}/opennn/pooling_layer.cpp
			${PLUGIN_DIR}/opennn/principal_components_layer.cpp
			${PLUGIN_DIR}/opennn/probabilistic_layer.cpp
			${PLUGIN_DIR}/opennn/pruning_inputs.cpp
			${PLUGIN_DIR}/opennn/quasi_newton_method.cpp
			${PLUGIN_DIR}/opennn/recurrent_layer.cpp
			${PLUGIN_DIR}/opennn/response_optimization.cpp
			${PLUGIN_DIR}/opennn/scaling_layer.cpp
			${PLUGIN_DIR}/opennn/statistics.cpp
			${PLUGIN_DIR}/opennn/stochastic_gradient_descent.cpp
			${PLUGIN_DIR}/opennn/sum_squared_error.cpp
#			${PLUGIN_DIR}/opennn/tensor_utilities.cpp
			${PLUGIN_DIR}/opennn/testing_analysis.cpp
			${PLUGIN_DIR}/opennn/tinyxml2.cpp
			${PLUGIN_DIR}/opennn/training_strategy.cpp
			${PLUGIN_DIR}/opennn/unit_testing.cpp
			${PLUGIN_DIR}/opennn/unscaling_layer.cpp
			${PLUGIN_DIR}/opennn/weighted_squared_error.cpp
)


##### PLUGIN HEADERS #####
list(APPEND PLUGIN_INCLUDES
			${PLUGIN_DIR}/src
			${PLUGIN_DIR}/opennn
			${PLUGIN_DIR}/eigen/Eigen/src
	
)


##### PLUGIN DEPENDENCY LOCATIONS (IF ANY) #####
if(WIN32)
	list(APPEND PLUGIN_LINKDIRS
				${PREBUILT_DIR}/lib/win32
				${PREBUILT_DIR}/bin
	)
	
	#set(PLUGIN_LIBDIRS ${PREBUILT_DIR}/lib/win32)
	list(APPEND PLUGIN_LIBDIRS
				${PREBUILT_DIR}/lib/win32
	)
elseif(LINUX)
	#Linux libs here
elseif(APPLE)
	#Macos libs here
endif(WIN32)


##### PLUGIN DEPENDENCY LINKING (IF ANY) #####					
if(WIN32)
	list(APPEND PLUGIN_LINKLIBS
				gomp
				#Lib names
	)
elseif(LINUX)
	list(APPEND PLUGIN_LINKLIBS
				#Lib names
	)
elseif(APPLE)
	list(APPEND PLUGIN_LINKLIBS
				#Lib names
	)
endif(WIN32)