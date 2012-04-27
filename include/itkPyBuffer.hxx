/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef __itkPyBuffer_hxx
#define __itkPyBuffer_hxx

#include "itkPyBuffer.h"

// Support NumPy < 1.7
#ifndef NPY_ARRAY_C_CONTIGUOUS
#define NPY_ARRAY_C_CONTIGUOUS NPY_C_CONTIGUOUS
#endif

#ifndef NPY_ARRAY_F_CONTIGUOUS
#define NPY_ARRAY_F_CONTIGUOUS NPY_F_CONTIGUOUS
#endif

#ifndef NPY_ARRAY_WRITEABLE
#define NPY_ARRAY_WRITEABLE NPY_WRITEABLE
#endif

namespace itk
{

template<class TImage>
PyObject *
PyBuffer<TImage>
::GetArrayFromImage( ImageType * image, bool keepAxes)
{
  if( !image )
  {
    throw std::runtime_error("Input image is null");
  }

  image->Update();

  ComponentType * buffer = const_cast < ComponentType * > ( image->GetBufferPointer() );
  char * data = (char *)( buffer );

  IndexType index;
  index.Fill(0);
  int nrOfComponents = DefaultConvertPixelTraits<PixelType>::GetNumberOfComponents(image->GetPixel(index));

  int item_type = PyTypeTraits<ComponentType>::value;

  int numpyArrayDimension = ( nrOfComponents > 1) ? ImageDimension + 1 : ImageDimension;

  // Construct array with dimensions
  npy_intp dimensions[ numpyArrayDimension ];

  // Add a dimension if there are more than one component
  if ( nrOfComponents > 1)
  {
    dimensions[0] = nrOfComponents;
  }
  int dimensionOffset = ( nrOfComponents > 1) ? 1 : 0;

  SizeType size = image->GetBufferedRegion().GetSize();
  for(unsigned int d=0; d < ImageDimension; d++ )
  {
    dimensions[d + dimensionOffset] = size[d];
  }

  if (!keepAxes)
  {
    // Reverse dimensions array
    npy_intp reverseDimensions[ numpyArrayDimension ];
    for(int d=0; d < numpyArrayDimension; d++ )
    {
        reverseDimensions[d] = dimensions[numpyArrayDimension - d - 1];
    }

    for(int d=0; d < numpyArrayDimension; d++ )
    {
        dimensions[d] = reverseDimensions[d];
    }
  }

  int flags = (keepAxes? NPY_ARRAY_F_CONTIGUOUS : NPY_ARRAY_C_CONTIGUOUS) |
              NPY_WRITEABLE;

  PyObject * obj = PyArray_New(&PyArray_Type, numpyArrayDimension, dimensions, item_type, NULL, data, 0, flags, NULL);

  return obj;
}

template<class TImage>
const typename PyBuffer<TImage>::OutImagePointer
PyBuffer<TImage>
::GetImageFromArray( PyObject *obj )
{

    int element_type = PyTypeTraits<ComponentType>::value;

    PyArrayObject * parray =
          (PyArrayObject *) PyArray_ContiguousFromObject(
                                                    obj,
                                                    element_type,
                                                    ImageDimension,
                                                    ImageDimension  );

    if( parray == NULL )
      {
      throw std::runtime_error("Contiguous array couldn't be created from input python object");
      }

    const unsigned int imageDimension = parray->nd;

    SizeType size;

    unsigned int numberOfPixels = 1;

    for( unsigned int d=0; d<imageDimension; d++ )
      {
      size[imageDimension - d - 1]         = parray->dimensions[d];
      numberOfPixels *= parray->dimensions[d];
      }

    IndexType start;
    start.Fill( 0 );

    RegionType region;
    region.SetIndex( start );
    region.SetSize( size );

    PointType origin;
    origin.Fill( 0.0 );

    SpacingType spacing;
    spacing.Fill( 1.0 );

    ImporterPointer importer = ImporterType::New();
    importer->SetRegion( region );
    importer->SetOrigin( origin );
    importer->SetSpacing( spacing );

    const bool importImageFilterWillOwnTheBuffer = false;

    ComponentType * data = (ComponentType *)parray->data;

    importer->SetImportPointer(
                        data,
                        numberOfPixels,
                        importImageFilterWillOwnTheBuffer );

    importer->Update();
    OutImagePointer output = importer->GetOutput();
    output->DisconnectPipeline();

    return output;
}


} // namespace itk

#endif
