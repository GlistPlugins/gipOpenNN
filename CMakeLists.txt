cmake_minimum_required (VERSION 3.10.2)

##### PLUGIN NAME #####
set(pluginname gipOpenNN)


##### PLUGIN PATHS #####
set(PLUGIN_DIR ${PLUGINS_DIR}/${pluginname})
set(PREBUILT_DIR ${PLUGIN_DIR}/prebuilts)


##### PLUGIN SOURCES #####
list(APPEND PLUGIN_SRCS
			${PLUGIN_DIR}/src/gipOpenNN.cpp
			${PLUGIN_DIR}/libraries/opennn/adaptive_moment_estimation.cpp
			${PLUGIN_DIR}/libraries/opennn/bounding_layer.cpp
			${PLUGIN_DIR}/libraries/opennn/conjugate_gradient.cpp
			${PLUGIN_DIR}/libraries/opennn/convolutional_layer.cpp
			${PLUGIN_DIR}/libraries/opennn/correlations.cpp
			${PLUGIN_DIR}/libraries/opennn/cross_entropy_error.cpp
			${PLUGIN_DIR}/libraries/opennn/data_set.cpp
			${PLUGIN_DIR}/libraries/opennn/genetic_algorithm.cpp
			${PLUGIN_DIR}/libraries/opennn/gradient_descent.cpp
			${PLUGIN_DIR}/libraries/opennn/growing_inputs.cpp
			${PLUGIN_DIR}/libraries/opennn/growing_neurons.cpp
			${PLUGIN_DIR}/libraries/opennn/inputs_selection.cpp
			${PLUGIN_DIR}/libraries/opennn/layer.cpp
			${PLUGIN_DIR}/libraries/opennn/learning_rate_algorithm.cpp
			${PLUGIN_DIR}/libraries/opennn/levenberg_marquardt_algorithm.cpp
			${PLUGIN_DIR}/libraries/opennn/long_short_term_memory_layer.cpp
			${PLUGIN_DIR}/libraries/opennn/loss_index.cpp
			${PLUGIN_DIR}/libraries/opennn/mean_squared_error.cpp
			${PLUGIN_DIR}/libraries/opennn/minkowski_error.cpp
			${PLUGIN_DIR}/libraries/opennn/model_selection.cpp
			${PLUGIN_DIR}/libraries/opennn/neural_network.cpp
			${PLUGIN_DIR}/libraries/opennn/neurons_selection.cpp
			${PLUGIN_DIR}/libraries/opennn/normalized_squared_error.cpp
			${PLUGIN_DIR}/libraries/opennn/numerical_differentiation.cpp
			${PLUGIN_DIR}/libraries/opennn/opennn_strings.cpp
			${PLUGIN_DIR}/libraries/opennn/optimization_algorithm.cpp
			${PLUGIN_DIR}/libraries/opennn/perceptron_layer.cpp
			${PLUGIN_DIR}/libraries/opennn/pooling_layer.cpp
			${PLUGIN_DIR}/libraries/opennn/principal_components_layer.cpp
			${PLUGIN_DIR}/libraries/opennn/probabilistic_layer.cpp
			${PLUGIN_DIR}/libraries/opennn/pruning_inputs.cpp
			${PLUGIN_DIR}/libraries/opennn/quasi_newton_method.cpp
			${PLUGIN_DIR}/libraries/opennn/recurrent_layer.cpp
			${PLUGIN_DIR}/libraries/opennn/response_optimization.cpp
			${PLUGIN_DIR}/libraries/opennn/scaling_layer.cpp
			${PLUGIN_DIR}/libraries/opennn/statistics.cpp
			${PLUGIN_DIR}/libraries/opennn/stochastic_gradient_descent.cpp
			${PLUGIN_DIR}/libraries/opennn/sum_squared_error.cpp
			${PLUGIN_DIR}/libraries/opennn/testing_analysis.cpp
			${PLUGIN_DIR}/libraries/opennn/tinyxml2.cpp
			${PLUGIN_DIR}/libraries/opennn/training_strategy.cpp
			${PLUGIN_DIR}/libraries/opennn/unit_testing.cpp
			${PLUGIN_DIR}/libraries/opennn/unscaling_layer.cpp
			${PLUGIN_DIR}/libraries/opennn/weighted_squared_error.cpp
)


##### PLUGIN HEADERS #####
list(APPEND PLUGIN_INCLUDES
			${PLUGIN_DIR}/src
			${PLUGIN_DIR}/libraries/opennn
			${PLUGIN_DIR}/libraries/eigen/Eigen/src
	
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
				gomp
	)
elseif(APPLE)
	list(APPEND PLUGIN_LINKLIBS
				#Lib names
	)
endif(WIN32)