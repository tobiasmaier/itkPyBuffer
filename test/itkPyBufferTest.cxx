#include <iostream>

#include "NumpyHelper.h"

#include "itkPyBuffer.h"
#include "itkImage.h"
#include "itkVectorImage.h"


int itkPyBufferTest(int, char * [])
{
    try
    {
        NumpyHelper::Initialize();

        const unsigned int Dimension = 3;
        typedef unsigned char                                  PixelType;
        typedef itk::Image<PixelType, Dimension>       ScalarImageType;
        typedef itk::VectorImage<PixelType, Dimension> VectorImageType;
        typedef itk::ImageRegion<Dimension>            RegionType;

        RegionType region;
        region.SetSize(0,200);
        region.SetSize(1,100);
        region.SetSize(2,10);
   
        // Test for scalar image
        ScalarImageType::Pointer scalarImage = ScalarImageType::New();
        scalarImage->SetRegions(region);
        scalarImage->Allocate();

        itk::PyBuffer<ScalarImageType>::GetArrayFromImage(scalarImage);

    }
    catch(itk::ExceptionObject &err)
    {
        (&err)->Print(std::cerr);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
