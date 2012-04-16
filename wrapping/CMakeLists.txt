if(ITK_WRAP_PYTHON)

  itk_wrap_module(ITKBufferConvertion)

  set(WRAPPER_SWIG_LIBRARY_FILES ${WRAPPER_SWIG_LIBRARY_FILES} "${CMAKE_CURRENT_SOURCE_DIR}/PyBuffer.i")
  
  itk_auto_load_submodules()
  itk_end_wrap_module()

  wrap_itk_bindings_install("/Python/itkExtras/" "${CMAKE_CURRENT_SOURCE_DIR}/PyBufferExtras.py")

endif(ITK_WRAP_PYTHON)
