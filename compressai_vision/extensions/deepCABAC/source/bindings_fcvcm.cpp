#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Lib/CommonLib/TypeDef.h>
#include <Lib/CommonLib/Quant.h>
#include <Lib/EncLib/CABACEncoder.h>
#include <Lib/DecLib/CABACDecoder.h>
#include <iostream>
#include <math.h>

namespace py = pybind11;

class Encoder
{
public:
  Encoder() { m_CABACEncoder.startCabacEncoding( &m_Bytestream ); }
  ~Encoder() {}
  void      initCtxModels( uint32_t cabac_unary_length_minus1, uint8_t param_opt_flag) { m_CABACEncoder.initCtxMdls(cabac_unary_length_minus1+1, param_opt_flag); }
  void      iae_v( uint8_t v, int32_t value )            { m_CABACEncoder.iae_v( v, value ); }
  void      uae_v( uint8_t v, uint32_t value )           { m_CABACEncoder.uae_v( v, value ); }

  int32_t   quantFeatures ( py::array_t<float32_t, py::array::c_style> Features, py::array_t<int32_t, py::array::c_style> qIndex, int32_t qpDensity, int32_t qp, int32_t scan_order );
  uint32_t  encodeFeatures( py::array_t<int32_t, py::array::c_style> qindex, int32_t scan_order  );
  py::array_t<uint8_t>  finish();

private:
  std::vector<uint8_t>  m_Bytestream;
  CABACEncoder          m_CABACEncoder;

};

int32_t Encoder::quantFeatures( py::array_t<float32_t, py::array::c_style> Features, py::array_t<int32_t, py::array::c_style> qIndex, int32_t qpDensity, int32_t qp, int32_t scan_order )
{
  uint32_t totalNumFeatures = 1;
  uint32_t layerWidth = 1;

  py::buffer_info features = Features.request();
  py::buffer_info quntizedIndex = qIndex.request();
  float32_t* pFeatures          = (float32_t*) features.ptr;
  int32_t* pQIndex = (int32_t*) quntizedIndex.ptr;

  for (size_t idx = 0; idx < (size_t)features.ndim; idx++)
  {
    totalNumFeatures *= features.shape[idx];
    if( idx == 0 ) { continue; }
    layerWidth *= features.shape[idx];
  }

  int32_t k = 1 << qpDensity;
  int32_t mul = k + (qp & (k-1));
  int32_t shift = qp >> qpDensity;
  float32_t qStepSize = mul * pow(2.0, shift - qpDensity);

  int32_t success = quantizeFeatures(pFeatures, pQIndex, qStepSize, layerWidth, totalNumFeatures, 0);

  return success;
}

uint32_t Encoder::encodeFeatures( py::array_t<int32_t, py::array::c_style> qindex, int32_t scan_order )
{
  py::buffer_info bi_qindex = qindex.request();
  int32_t* pQindex          = (int32_t*) bi_qindex.ptr;

  uint32_t layerWidth = 1;
  uint32_t numWeights = 1;
  for( size_t idx = 0; idx < (size_t)bi_qindex.ndim; idx++ )
  {
    numWeights *= bi_qindex.shape[idx];
    if( idx == 0 ) { continue; }
    layerWidth *= bi_qindex.shape[idx];
  }

  return m_CABACEncoder.encodeWeights(pQindex, layerWidth, numWeights, 0, scan_order);
}

py::array_t<uint8_t> Encoder::finish()
{
  m_CABACEncoder.terminateCabacEncoding();

  auto Result = py::array_t<uint8_t, py::array::c_style>(m_Bytestream.size());
  py::buffer_info bi_Result = Result.request();
  uint8_t* pResult = (uint8_t*) bi_Result.ptr;

  for( size_t idx = 0; idx < m_Bytestream.size(); idx ++ )
  {
    pResult[idx] = m_Bytestream.at(idx);
  }
  return Result;
}


// Decoder

class Decoder
{
public:
  Decoder() {}
  ~Decoder() {}
  void     setStream        ( py::array_t<uint8_t, py::array::c_style> Bytestream );
  void     initCtxModels    ( uint32_t cabac_unary_length_minus1 ) { m_CABACDecoder.initCtxMdls( cabac_unary_length_minus1+1 ); }
  int32_t  iae_v            ( uint8_t v ) { return m_CABACDecoder.iae_v(v); }
  uint32_t uae_v            ( uint8_t v ) { return m_CABACDecoder.uae_v( v ); }

  void     decodeFeatures   ( py::array_t<int32_t, py::array::c_style> Features, int32_t scan_order );
  void     dequantFeatures  ( py::array_t<float32_t, py::array::c_style> Features, py::array_t<int32_t, py::array::c_style> qIndex, int32_t qpDensity, int32_t qp, int32_t scan_order);
  uint32_t finish       ();

private:
  CABACDecoder  m_CABACDecoder;
};

void Decoder::setStream( py::array_t<uint8_t, py::array::c_style> Bytestream )
{
  py::buffer_info bi_Bytestream = Bytestream.request();
  uint8_t* pBytestream          = (uint8_t*) bi_Bytestream.ptr;
  m_CABACDecoder.startCabacDecoding( pBytestream );
}

void Decoder::decodeFeatures( py::array_t<int32_t, py::array::c_style> Features, int32_t scan_order )    
{
  uint32_t layerWidth = 1;
  uint32_t totalNumFeatures = 1;

  py::buffer_info bi_Features = Features.request();
  int32_t* pFeatures   = (int32_t*) bi_Features.ptr;



  for( size_t idx = 0; idx < (size_t)bi_Features.ndim; idx++ )
  {
    totalNumFeatures *= bi_Features.shape[idx];
    if( idx == 0 ) { continue; }
    layerWidth *= bi_Features.shape[idx];
  }

  m_CABACDecoder.decodeWeights(pFeatures, layerWidth, totalNumFeatures, 0, scan_order);
}


void Decoder::dequantFeatures(py::array_t<float32_t, py::array::c_style> Features, py::array_t<int32_t, py::array::c_style> qIndex, int32_t qpDensity, int32_t qp, int32_t scan_order)
{
  uint32_t totalNumFeatures = 1;
  uint32_t layerWidth = 1;

  py::buffer_info features = Features.request();
  py::buffer_info quntizedIndex = qIndex.request();
  float32_t* pFeatures          = (float32_t*) features.ptr;
  int32_t* pQIndex = (int32_t*) quntizedIndex.ptr;

  for (size_t idx = 0; idx < (size_t)features.ndim; idx++)
  {
    totalNumFeatures *= features.shape[idx];
    if( idx == 0 ) { continue; }
    layerWidth *= features.shape[idx];
  }

  int32_t k = 1 << qpDensity;
  int32_t mul = k + (qp & (k-1));
  int32_t shift = qp >> qpDensity;
  float32_t qStepSize = mul * pow(2.0, shift - qpDensity);

  deQuantize(pFeatures, pQIndex, qStepSize, totalNumFeatures, layerWidth, 0);
}

uint32_t Decoder::finish()
{
  uint32_t bytesRead = m_CABACDecoder.terminateCabacDecoding();
  return bytesRead;
}

PYBIND11_MODULE(fcvcmCABAC, m) 
{
    py::class_<Encoder>(m, "Encoder")
        .def( py::init<>())
        .def( "iae_v",         &Encoder::iae_v         )
        .def( "uae_v",         &Encoder::uae_v         )
        .def( "initCtxModels" , &Encoder::initCtxModels   )
        .def( "quantFeatures" , &Encoder::quantFeatures   )
        .def( "encodeFeatures", &Encoder::encodeFeatures  )
        .def( "finish"        , &Encoder::finish          );

    py::class_<Decoder>(m, "Decoder")
      .def( py::init<>())
      .def( "setStream",        &Decoder::setStream, py::keep_alive<1, 2>() )
      .def( "initCtxModels",    &Decoder::initCtxModels )
      .def( "iae_v",            &Decoder::iae_v         )
      .def( "uae_v",            &Decoder::uae_v         )
      .def( "dequantFeatures",  &Decoder::dequantFeatures  )
      .def( "decodeFeatures",   &Decoder::decodeFeatures   )
      .def( "finish",           &Decoder::finish        );

}