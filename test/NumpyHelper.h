#include <Python.h>
#include <arrayobject.h>

class NumpyHelper
{
public:
    static void Initialize()
    {
        if ( instance == NULL )
            instance = new NumpyHelper();
    }

protected:
    NumpyHelper() {
        Py_Initialize();
        import_array();
    }

    ~NumpyHelper()
    {
        Py_Finalize();
    }

private:
    static NumpyHelper* instance;;
};
NumpyHelper* NumpyHelper::instance = NULL;
