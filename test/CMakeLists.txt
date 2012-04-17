itk_module_test()

set(ITKBufferConvertionTests
itkPyBufferTest.cxx
)

set(ITKBufferConvertion-Test_LIBRARIES
    "${ITKBufferConvertion-Test_LIBRARIES};${PYTHON_LIBRARY}")

CreateTestDriver(ITKBufferConvertion  "${ITKBufferConvertion-Test_LIBRARIES}"
    "${ITKBufferConvertionTests}")

itk_add_test(NAME itkPyBufferTest
    COMMAND ITKBufferConvertionTestDriver itkPyBufferTest)

