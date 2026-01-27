#pragma once

#ifndef MEMORY_CLASSES_H
#define MEMORY_CLASSES_H

#include <new>          // std::bad_alloc
#include <stdexcept>    // std::runtime_error
#include <cstdlib>      // malloc/calloc/free


/** Tag dispatch namespace */
namespace MemoryTag {
    struct Zeroed { };      // Dispatch to zeroed-memory constructor overloads
};


template <class DataT, class DeleterT>
class UniquePointer {
    DataT *data;

protected:
    UniquePointer(): data(NULL) { }
    UniquePointer(DataT *ptr): data(ptr) { }

public:
    virtual ~UniquePointer() { DeleterT()(data); }

    /** Get a non-owning instance of the stored pointer */
    DataT *get() { return data; }

    /** Release ownership of the stored pointer */
    DataT *release() {

        DataT *res = data;

        data = NULL;
        return res;
    }

    DataT *operator->() { return data; }
    const DataT *operator->() const { return data; }

    DataT &operator*() { return *data; }
    const DataT &operator*() const { return *data; }

    DataT &operator[](size_t idx) { return data[idx]; }
    const DataT &operator[](size_t idx) const { return data[idx]; }

    operator DataT *() { return data; }
    operator const DataT *() const { return data; }

    operator bool() const { return data != NULL; }
};


struct HostDeleter {
    void operator()(void *ptr) { free(ptr); }
};

/** A smart pointer that manages memory with malloc/free */
template <class DataT>
struct HostPointer: public UniquePointer<DataT, HostDeleter> {

    /** Allocate space for @p count objects */
    explicit HostPointer(size_t count = 1):
        UniquePointer<DataT, HostDeleter>((DataT *)malloc(sizeof (DataT) * count))
    {
        if (!this->get() && count) {
            throw std::bad_alloc();
        }
    }

    /** Allocate space for @p count objects, with their memory zeroed */
    HostPointer(const MemoryTag::Zeroed &tag, size_t count = 1):
        UniquePointer<DataT, HostDeleter>((DataT *)calloc(count, sizeof (DataT)))
    {
        (void)tag;
        if (!this->get() && count) {
            throw std::bad_alloc();
        }
    }
};


struct CudaDeleter {
    void operator()(void *ptr) { cudaFree(ptr); }
};

/** A smart pointer that manages memory with cudaMalloc/cudaFree */
template <class DataT>
class DevicePointer: public UniquePointer<DataT, CudaDeleter> {

    void throw_if_bad(cudaError_t err) {

        if (err) {
            throw std::runtime_error(cudaGetErrorString(err));
        }

    }

    DataT *allocate(size_t count) {

        cudaError_t err;
        void *res;

        err = cudaMalloc(&res, sizeof (DataT) * count);
        throw_if_bad(err);
        return static_cast<DataT *>(res);

    }

public:
    /** Allocate space for @p count objects */
    explicit DevicePointer(size_t count):
        UniquePointer<DataT, CudaDeleter>(allocate(count)) { }

    /** Allocate zeroed memory for @p count objects */
    DevicePointer(const MemoryTag::Zeroed &tag, size_t count):
        UniquePointer<DataT, CudaDeleter>(allocate(count))
    {
        cudaError_t err;

        (void)tag;
        err = cudaMemset(this->get(), 0, sizeof (DataT) * count);
        throw_if_bad(err);
    }

    /** Copy a buffer of host memory into a new buffer on the device */
    DevicePointer(const DataT buf[], size_t count = 1):
        UniquePointer<DataT, CudaDeleter>(allocate(count))
    {
        cudaError_t err;

        err = cudaMemcpy(this->get(), buf, sizeof (DataT) * count, cudaMemcpyHostToDevice);
        throw_if_bad(err);
    }
};


#endif /* MEMORY_CLASSES_H */
