/* Copyright (c) 2022, NVIDIA CORPORATION.

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* The Filigree Python C API extension. This provides access to various methods
 * of a Filgree Image object in Python. The interface it provides is quite
 * low-level, and is wrapped by api.py to provide a more Pythonic interface.
 */

#define PY_SSIZE_T_CLEAN
#include "filigree.hxx"
#include "rmm/mr/device/per_device_resource.hpp"
#include <Python.h>
#include <structmember.h>

/* We need a stream for RMM. Since we don't make use of streams for asynchronous
 * operation throughout the examples, creating a single stream in this wrapper
 * suffices. */
static rmm::cuda_stream image_stream;

/* ImageProxy holds a pointer to a Filigree Image object inside a Python object.
 */
struct ImageProxy {
  PyObject_HEAD filigree::Image *image;
  bool valid;

  ImageProxy() : image(nullptr), valid(false) {}
};

/* Create a new Filigree Image object from a given file. */
static int Image_init(ImageProxy *self, PyObject *args, PyObject *kwds) {
  const char *filename;
  if (!PyArg_ParseTuple(args, "s", &filename)) {
    return -1;
  }

  try {
    self->image = new filigree::Image(
        filename, rmm::mr::get_current_device_resource(), image_stream);
    self->valid = true;
  } catch (const std::exception &e) {
    self->valid = false;
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return -1;
  }

  return 0;
}

/* Deallocate an image. We must only call the Image dtor if a valid Image object
 * is held - a valid Image may not be held if this deallocation is called due to
 * an error initializing the Image. */
static void Image_dealloc(ImageProxy *self) {
  if (self->valid) {
    self->image->~Image();
  }

  self->~ImageProxy();
}

/* Get the Image data and provide it in a Python buffer object. */
static PyObject *Image_get_data(ImageProxy *self, PyObject *args,
                                PyObject *kws) {
  Magick::PixelPacket *pixels;

  try {
    pixels = self->image->get_data();
  } catch (const std::exception &e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }

  PyObject *data = PyBytes_FromStringAndSize(
      reinterpret_cast<const char *>(pixels), self->image->get_alloc_size());

  delete[] pixels;

  return data;
}

/* Return a pointer to the image data as a Python int. */
static PyObject *Image_get_pixel_ptr(ImageProxy *self, PyObject *args,
                                     PyObject *kws) {
  return PyLong_FromVoidPtr(self->image->get_pixel_ptr());
}

/* Return the image width. */
static PyObject *Image_get_width(ImageProxy *self, PyObject *args,
                                 PyObject *kws) {
  return PyLong_FromSize_t(self->image->get_width());
}

/* Return the image height. */
static PyObject *Image_get_height(ImageProxy *self, PyObject *args,
                                  PyObject *kws) {
  return PyLong_FromSize_t(self->image->get_height());
}

/* Convert the image to greyscale using the built-in Filigree kernel. */
static PyObject *Image_apply_weighted_greyscale(ImageProxy *self,
                                                PyObject *args, PyObject *kws) {
  self->image->to_greyscale();
  Py_RETURN_NONE;
}

/* Convert the image to black and white using the built-in Filigree kernel. */
static PyObject *Image_binarize(ImageProxy *self, PyObject *args,
                                PyObject *kws) {
  self->image->to_binary();
  Py_RETURN_NONE;
}

static PyMethodDef Image_methods[] = {
    {"get_data", (PyCFunction)Image_get_data, METH_NOARGS, NULL},
    {"get_pixel_ptr", (PyCFunction)Image_get_pixel_ptr, METH_NOARGS, NULL},
    {"get_width", (PyCFunction)Image_get_width, METH_NOARGS, NULL},
    {"get_height", (PyCFunction)Image_get_height, METH_NOARGS, NULL},
    {"apply_weighted_greyscale", (PyCFunction)Image_apply_weighted_greyscale,
     METH_NOARGS, NULL},
    {"binarize", (PyCFunction)Image_binarize, METH_NOARGS, NULL},
    {NULL},
};

static PyMemberDef Image_members[] = {
    {(char *)"valid", T_BOOL, offsetof(ImageProxy, valid), 0, NULL},
    {NULL} /* Sentinel */
};

static PyTypeObject ImageType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "filigree._lib", /* tp_name */
    sizeof(ImageProxy),                                /* tp_basicsize */
    0,                                                 /* tp_itemsize */
    (destructor)Image_dealloc,                         /* tp_dealloc */
    0,                                                 /* tp_print */
    0,                                                 /* tp_getattr */
    0,                                                 /* tp_setattr */
    0,                                                 /* tp_compare */
    0,                                                 /* tp_repr */
    0,                                                 /* tp_as_number */
    0,                                                 /* tp_as_sequence */
    0,                                                 /* tp_as_mapping */
    0,                                                 /* tp_hash */
    0,                                                 /* tp_call*/
    0,                                                 /* tp_str*/
    0,                                                 /* tp_getattro*/
    0,                                                 /* tp_setattro*/
    0,                                                 /* tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,                                /* tp_flags*/
    "On-device image object",                          /* tp_doc */
    0,                                                 /* tp_traverse */
    0,                                                 /* tp_clear */
    0,                                                 /* tp_richcompare */
    0,                                                 /* tp_weaklistoffset */
    0,                                                 /* tp_iter */
    0,                                                 /* tp_iternext */
    Image_methods,                                     /* tp_methods */
    Image_members,                                     /* tp_members */
    0,                                                 /* tp_getset */
    0,                                                 /* tp_base */
    0,                                                 /* tp_dict */
    0,                                                 /* tp_descr_get */
    0,                                                 /* tp_descr_set */
    0,                                                 /* tp_dictoffset */
    (initproc)Image_init,                              /* tp_init */
    0,                                                 /* tp_alloc */
    0,                                                 /* tp_new */
    0,                                                 /* tp_free */
    0,                                                 /* tp_is_gc */
    0,                                                 /* tp_bases */
    0,                                                 /* tp_mro */
    0,                                                 /* tp_cache */
    0,                                                 /* tp_subclasses */
    0,                                                 /* tp_weaklist */
    0,                                                 /* tp_del */
    0,                                                 /* tp_version_tag */
    0,                                                 /* tp_finalize */
#if PY_MAJOR_VERSION == 3
/* Python 3.8 has two slots, 3.9 has one. */
#if PY_MINOR_VERSION > 7
    0, /* tp_vectorcall */
#if PY_MINOR_VERSION == 8
    0, /* tp_print */
#endif
#endif
#endif
};

/* Get the ImageMagick Quantum depth. This is generally expected to be 16, but
 * this method can be used to check. */
static PyObject *get_quantum_depth(PyObject *self, PyObject *args) {
  return PyLong_FromLong(MAGICKCORE_QUANTUM_DEPTH);
}

static PyMethodDef ext_methods[] = {{"get_quantum_depth", get_quantum_depth,
                                     METH_NOARGS,
                                     "Return the ImageMagick Quantum depth"},
                                    {nullptr}};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_lib",
    "A simple customisable image processing library for CUDA", -1, ext_methods};

PyMODINIT_FUNC PyInit__lib(void) {
  PyObject *m = PyModule_Create(&moduledef);
  if (m == nullptr) {
    return nullptr;
  }

  ImageType.tp_new = PyType_GenericNew;
  if (PyType_Ready(&ImageType) < 0) {
    return nullptr;
  }

  Py_INCREF(&ImageType);
  PyModule_AddObject(m, "Image", (PyObject *)(&ImageType));

  return m;
}
