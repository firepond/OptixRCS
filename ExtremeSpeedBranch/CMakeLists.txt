set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/reduceKernels.cu
  PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ
)

OPTIX_add_sample_executable( ExtremeSpeedBranch target_name
  tiny_obj_loader.h

  rcs_params.h
  rcs_predicitor.h
  main.cpp
  
  triangles_rcs.cu
  complex_vector.cu

  reduceKernels.cu
  reduceKernels.h

  # OPTIONS -rdc true
  )

target_link_libraries( ${target_name}
  ${CUDA_LIBRARIES}
  )
