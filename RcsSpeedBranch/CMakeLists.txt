OPTIX_add_sample_executable( RcsSpeedBranch target_name
  tiny_obj_loader.h

  rcs_params.h
  triangles_rcs.h
  main.cpp
  
  triangles_rcs.cu
  complex_vector.cu

  reduce.cu

  OPTIONS -rdc true
  )

target_link_libraries( ${target_name}
  ${CUDA_LIBRARIES}
  )
