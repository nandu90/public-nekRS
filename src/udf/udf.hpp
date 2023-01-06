#if !defined(nekrs_udf_hpp_)
#define nekrs_udf_hpp_

#define ins_t nrs_t

#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"
#include "parReader.hpp"
#include "constantFlowRate.hpp"
#include "postProcessing.hpp"
#include "plugins/velRecycling.hpp"
#include "plugins/tavg.hpp"
#include "plugins/lowMach.hpp"
#include "plugins/RANSktau.hpp"

#define PASS { if (platform->comm.mpiRank == 0) printf("TESTS passed \n"); platform->exitValue = 0;}
#define FAIL { if (platform->comm.mpiRank == 0) printf("TESTS failed!\n"); platform->exitValue = 1;}

extern "C" {
void UDF_Setup0(MPI_Comm comm, setupAide &options);
void UDF_Setup(nrs_t* nrs);
void UDF_LoadKernels(occa::properties& kernelInfo);
void UDF_AutoLoadKernels(occa::properties& kernelInfo);
void UDF_ExecuteStep(nrs_t* nrs, dfloat time, int tstep);
}

typedef void (* udfsetup0)(MPI_Comm comm, setupAide &options);
typedef void (* udfsetup)(nrs_t* nrs);
typedef void (* udfloadKernels)(occa::properties& kernelInfo);
typedef void (* udfautoloadKernels)(occa::properties& kernelInfo);
typedef void (* udfexecuteStep)(nrs_t* nrs, dfloat time, int tstep);

typedef void (* udfuEqnSource)(nrs_t* nrs, dfloat time, occa::memory o_U, occa::memory o_FU);
typedef void (* udfsEqnSource)(nrs_t* nrs, dfloat time, occa::memory o_S, occa::memory o_SU);
typedef void (* udfproperties)(nrs_t* nrs, dfloat time, occa::memory o_U,
                               occa::memory o_S, occa::memory o_UProp,
                               occa::memory o_SProp);
typedef void (* udfdiv)(nrs_t* nrs, dfloat time, occa::memory o_div);
typedef int (* udfconv)(nrs_t* nrs, int stage);

struct UDF
{
  udfsetup0 setup0;
  udfsetup setup;
  udfloadKernels loadKernels;
  udfautoloadKernels autoloadKernels;
  udfexecuteStep executeStep;
  udfuEqnSource uEqnSource;
  udfsEqnSource sEqnSource;
  udfproperties properties;
  udfdiv div;
  udfconv timeStepConverged;
};

extern UDF udf;

void oudfFindDirichlet(std::string &field);
void oudfFindNeumann(std::string &field);
void oudfInit(setupAide &options);
void udfBuild(const char* udfFile, setupAide& options);
void udfLoad(void);
void* udfLoadFunction(const char* fname, int errchk);
occa::kernel oudfBuildKernel(occa::properties kernelInfo, const char *function);

extern "C" {
#ifdef USE_AUTOLOADKERNEL 
#include "udfAutoLoadKernel.hpp"
#endif
}

#endif
