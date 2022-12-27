#include <vector>
#include <map>
#include <cctype>

#include "nrs.hpp"
#include "meshSetup.hpp"
#include "bdry.hpp"
#include "bcMap.hpp"
#include "nekInterfaceAdapter.hpp"
#include "udf.hpp"
#include "filter.hpp"
#include "avm.hpp"

#include "cdsSetup.cpp"
#include "parseMultigridSchedule.hpp"
#include <algorithm>

std::vector<int> determineMGLevels(std::string section)
{
  const std::string optionsPrefix = [section]() {
    std::string prefix = section + std::string(" ");
    if (section.find("temperature") != std::string::npos) {
      prefix = std::string("scalar00 ");
    }
    std::transform(prefix.begin(), prefix.end(), prefix.begin(), [](unsigned char c) {
      return std::toupper(c);
    });
    return prefix;
  }();

  std::vector<int> levels;
  int N;
  platform->options.getArgs("POLYNOMIAL DEGREE", N);

  std::string p_mgschedule = platform->options.getArgs(optionsPrefix + "MULTIGRID SCHEDULE");
  if(!p_mgschedule.empty()){

    // note: default order is not required here.
    // We just need the levels, not the degree.
    auto [scheduleMap, errorString] = parseMultigridSchedule(p_mgschedule, platform->options, 3);
    for(auto && [cyclePosition, smootherOrder] : scheduleMap){
      auto [order, isDownLeg] = cyclePosition;
      if(isDownLeg){
        levels.push_back(order);
      }
    }

    std::sort(levels.rbegin(), levels.rend());

    if (levels.back() > 1) {
      if (platform->options.compareArgs(optionsPrefix + "MULTIGRID COARSE SOLVE", "TRUE")) {
        // if the coarse level has p > 1 and requires solving the coarsest level,
        // rather than just smoothing, SEMFEM must be used for the discretization
        const auto usesSEMFEM =
            platform->options.compareArgs(optionsPrefix + "MULTIGRID SEMFEM", "TRUE");

        if (!usesSEMFEM) {
          if (platform->comm.mpiRank == 0) {
            printf("Error! FEM coarse discretization only supports p=1 for the coarsest level!\n");
          }
          ABORT(1);
        }
      }
    }

    return levels;
  }

  if (platform->options.compareArgs(optionsPrefix + "MULTIGRID SMOOTHER", "ASM") ||
           platform->options.compareArgs(optionsPrefix + "MULTIGRID SMOOTHER", "RAS")) {
    std::map<int, std::vector<int>> mg_level_lookup = {
        {1, {1}},
        {2, {2, 1}},
        {3, {3, 1}},
        {4, {4, 2, 1}},
        {5, {5, 3, 1}},
        {6, {6, 3, 1}},
        {7, {7, 3, 1}},
        {8, {8, 5, 1}},
        {9, {9, 5, 1}},
        {10, {10, 6, 1}},
        {11, {11, 6, 1}},
        {12, {12, 7, 1}},
        {13, {13, 7, 1}},
        {14, {14, 8, 1}},
        {15, {15, 9, 1}},
    };

    return mg_level_lookup.at(N);
  }

  std::map<int, std::vector<int>> mg_level_lookup = {
      {1, {1}},
      {2, {2, 1}},
      {3, {3, 1}},
      {4, {4, 2, 1}},
      {5, {5, 3, 1}},
      {6, {6, 4, 2, 1}},
      {7, {7, 5, 3, 1}},
      {8, {8, 6, 4, 1}},
      {9, {9, 7, 5, 1}},
      {10, {10, 8, 5, 1}},
      {11, {11, 9, 5, 1}},
      {12, {12, 10, 5, 1}},
      {13, {13, 11, 5, 1}},
      {14, {14, 12, 5, 1}},
      {15, {15, 13, 5, 1}},
  };

  return mg_level_lookup.at(N);
}

void printICMinMax(nrs_t *nrs)
{
  if(platform->comm.mpiRank == 0) 
    printf("================= INITITAL CONDITION ====================\n");

  {
    auto mesh = nrs->meshV;
    auto o_ux = nrs->o_U + 0*nrs->fieldOffset*sizeof(dfloat);
    auto o_uy = nrs->o_U + 1*nrs->fieldOffset*sizeof(dfloat);
    auto o_uz = nrs->o_U + 2*nrs->fieldOffset*sizeof(dfloat);
    const auto uxMin = platform->linAlg->min(mesh->Nlocal, o_ux, platform->comm.mpiComm);
    const auto uyMin = platform->linAlg->min(mesh->Nlocal, o_uy, platform->comm.mpiComm);
    const auto uzMin = platform->linAlg->min(mesh->Nlocal, o_uz, platform->comm.mpiComm);
    const auto uxMax = platform->linAlg->max(mesh->Nlocal, o_ux, platform->comm.mpiComm);
    const auto uyMax = platform->linAlg->max(mesh->Nlocal, o_uy, platform->comm.mpiComm);
    const auto uzMax = platform->linAlg->max(mesh->Nlocal, o_uz, platform->comm.mpiComm);
    if(platform->comm.mpiRank == 0) 
      printf("U min/max: %g %g %g %g %g %g\n", uxMin, uxMax, uyMin, uyMax, uzMin, uzMax);
  }

  {
    auto mesh = nrs->meshV;
    const auto prMin = platform->linAlg->min(mesh->Nlocal, nrs->o_P, platform->comm.mpiComm);
    const auto prMax = platform->linAlg->max(mesh->Nlocal, nrs->o_P, platform->comm.mpiComm);
    if(platform->comm.mpiRank == 0) 
      printf("P min/max: %g %g\n", prMin, prMax);
  }

  if (nrs->Nscalar) {
    auto cds = nrs->cds;
    if(platform->comm.mpiRank == 0) 
      printf("S min/max:");  
    for (int is = 0; is < cds->NSfields; is++) {
      if (!cds->compute[is])
        continue;

      mesh_t *mesh;
      (is) ? mesh = cds->meshV : mesh = cds->mesh[0]; // only first scalar can be a CHT mesh

      auto o_si = nrs->cds->o_S + nrs->cds->fieldOffset[is]; 
      const auto siMin = platform->linAlg->min(mesh->Nlocal, o_si, platform->comm.mpiComm);
      const auto siMax = platform->linAlg->max(mesh->Nlocal, o_si, platform->comm.mpiComm);
      if (platform->comm.mpiRank == 0) 
        printf(" %g %g", siMin, siMax);
    }
    if(platform->comm.mpiRank == 0) 
      printf("\n");  
  }
}

void nrsSetup(MPI_Comm comm, setupAide &options, nrs_t *nrs)
{
  {
    int N;
    platform->options.getArgs("POLYNOMIAL DEGREE", N);
    const int Nq = N + 1;
    if (BLOCKSIZE < Nq * Nq) {
      if (platform->comm.mpiRank == 0)
        printf("ERROR: several kernels requires BLOCKSIZE >= Nq * Nq."
               "BLOCKSIZE = %d, Nq*Nq = %d\n",
               BLOCKSIZE,
               Nq * Nq);
      ABORT(EXIT_FAILURE);
    }
  }
  platform_t *platform = platform_t::getInstance();
  device_t &device = platform->device;
  nrs->kernelInfo = new occa::properties();
  *(nrs->kernelInfo) = platform->kernelInfo;
  occa::properties &kernelInfo = *nrs->kernelInfo;
  kernelInfo["defines"].asObject();
  kernelInfo["includes"].asArray();
  kernelInfo["header"].asArray();
  kernelInfo["flags"].asObject();
  kernelInfo["include_paths"].asArray();

  int N, cubN;
  platform->options.getArgs("POLYNOMIAL DEGREE", N);
  platform->options.getArgs("CUBATURE POLYNOMIAL DEGREE", cubN);
  platform->options.getArgs("NUMBER OF SCALARS", nrs->Nscalar);
  platform->options.getArgs("MESH DIMENSION", nrs->dim);
  platform->options.getArgs("ELEMENT TYPE", nrs->elementType);
  if (platform->device.mode() == "Serial")
    platform->options.setArgs("GS OVERLAP", "FALSE");

  nrs->flow = 1;
  if (platform->options.compareArgs("VELOCITY SOLVER", "NONE"))
    nrs->flow = 0;

  if (nrs->flow) {
    if (platform->options.compareArgs("VELOCITY STRESSFORMULATION", "TRUE"))
      platform->options.setArgs("VELOCITY BLOCK SOLVER", "TRUE");
  }

  // init nek
  {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    std::string casename;
    platform->options.getArgs("CASENAME", casename);

    nek::setup(nrs);
    nek::setic();
    nek::userchk();
    if (platform->comm.mpiRank == 0)
      std::cout << "\n";
  }

  nrs->cht = 0;
  if (nekData.nelv != nekData.nelt && nrs->Nscalar)
    nrs->cht = 1;
  if (nrs->cht && !platform->options.compareArgs("SCALAR00 IS TEMPERATURE", "TRUE")) {
    if (platform->comm.mpiRank == 0)
      std::cout << "Conjugate heat transfer requires solving for temperature!\n";
    ABORT(EXIT_FAILURE);
    ;
  }
  if (nrs->cht && options.compareArgs("MOVING MESH", "TRUE")) {
    if (platform->comm.mpiRank == 0){
      std::cout << "Conjugate heat transfer + moving mesh is not supported\n";
    }
    ABORT(EXIT_FAILURE);
  }

  nrs->_mesh = createMesh(comm, N, cubN, nrs->cht, kernelInfo);
  nrs->meshV = (mesh_t *)nrs->_mesh->fluid;
  mesh_t *mesh = nrs->meshV;

  nrs->NVfields = 3;
  nrs->NTfields = nrs->NVfields + 1; // Total Velocity + Pressure
  mesh->Nfields = 1;

  platform->options.getArgs("SUBCYCLING STEPS", nrs->Nsubsteps);
  platform->options.getArgs("DT", nrs->dt[0]);

  if (platform->options.compareArgs("TIME INTEGRATOR", "TOMBO1")) {
    nrs->nBDF = 1;
  }
  else if (platform->options.compareArgs("TIME INTEGRATOR", "TOMBO2")) {
    nrs->nBDF = 2;
  }
  else if (platform->options.compareArgs("TIME INTEGRATOR", "TOMBO3")) {
    nrs->nBDF = 3;
  }
  nrs->nEXT = 3;
  if (nrs->Nsubsteps)
    nrs->nEXT = nrs->nBDF;
  nrs->coeffEXT = (dfloat *)calloc(nrs->nEXT, sizeof(dfloat));
  nrs->coeffBDF = (dfloat *)calloc(nrs->nBDF, sizeof(dfloat));

  nrs->nRK = 4;
  nrs->coeffSubEXT = (dfloat *)calloc(3, sizeof(dfloat));

  dfloat mue = 1;
  dfloat rho = 1;
  platform->options.getArgs("VISCOSITY", mue);
  platform->options.getArgs("DENSITY", rho);

  const dlong Nlocal = mesh->Nlocal;

  { // setup fieldOffset
    nrs->fieldOffset = mesh->Np * (mesh->Nelements + mesh->totalHaloPairs);
    mesh_t *meshT = nrs->_mesh;
    nrs->fieldOffset = mymax(nrs->fieldOffset, meshT->Np * (meshT->Nelements + meshT->totalHaloPairs));

    const int pageW = ALIGN_SIZE / sizeof(dfloat);
    if (nrs->fieldOffset % pageW)
      nrs->fieldOffset = (nrs->fieldOffset / pageW + 1) * pageW;
  }

  nrs->_mesh->fieldOffset = nrs->fieldOffset;

  { // setup cubatureOffset
    if (platform->options.compareArgs("ADVECTION TYPE", "CUBATURE")) {
      nrs->cubatureOffset = std::max(nrs->fieldOffset, mesh->Nelements * mesh->cubNp);
    }
    else {
      nrs->cubatureOffset = nrs->fieldOffset;
    }
    const int pageW = ALIGN_SIZE / sizeof(dfloat);
    if (nrs->cubatureOffset % pageW)
      nrs->cubatureOffset = (nrs->cubatureOffset / pageW + 1) * pageW;
  }

  if (nrs->Nsubsteps) {
    int Sorder;
    platform->options.getArgs("SUBCYCLING TIME ORDER", Sorder);
    if (Sorder == 4 && nrs->nRK == 4) { // ERK(4,4)
      dfloat rka[4] = {0.0, 1.0 / 2.0, 1.0 / 2.0, 1.0};
      dfloat rkb[4] = {1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0};
      dfloat rkc[4] = {0.0, 1.0 / 2.0, 1.0 / 2.0, 1.0};
      nrs->coeffsfRK = (dfloat *)calloc(nrs->nRK, sizeof(dfloat));
      nrs->weightsRK = (dfloat *)calloc(nrs->nRK, sizeof(dfloat));
      nrs->nodesRK = (dfloat *)calloc(nrs->nRK, sizeof(dfloat));
      memcpy(nrs->coeffsfRK, rka, nrs->nRK * sizeof(dfloat));
      memcpy(nrs->weightsRK, rkb, nrs->nRK * sizeof(dfloat));
      memcpy(nrs->nodesRK, rkc, nrs->nRK * sizeof(dfloat));
    }
    else {
      if (platform->comm.mpiRank == 0)
        std::cout << "Unsupported subcycling scheme!\n";
      ABORT(1);
    }
    nrs->o_coeffsfRK = device.malloc(nrs->nRK * sizeof(dfloat), nrs->coeffsfRK);
    nrs->o_weightsRK = device.malloc(nrs->nRK * sizeof(dfloat), nrs->weightsRK);
  }

  // setup mempool
  int ellipticMaxFields = 1;
  if (platform->options.compareArgs("VELOCITY BLOCK SOLVER", "TRUE"))
    ellipticMaxFields = nrs->NVfields;
  const int ellipticWrkFields = elliptic_t::NScratchFields * ellipticMaxFields;

  int wrkFields = 10;
  if (nrs->Nsubsteps)
    wrkFields = 9 + 3 * nrs->NVfields;
  if (options.compareArgs("MOVING MESH", "TRUE"))
    wrkFields += nrs->NVfields;

  const int mempoolNflds = std::max(wrkFields, 2 * nrs->NVfields + ellipticWrkFields);
  platform->create_mempool(nrs->fieldOffset, mempoolNflds);

  // offset mempool available for elliptic because also used it for ellipticSolve input/output
  auto const o_mempoolElliptic =
      platform->o_mempool.o_ptr.slice((2 * nrs->NVfields * sizeof(dfloat)) * nrs->fieldOffset);

  if (options.compareArgs("MOVING MESH", "TRUE")) {
    const int nBDF = std::max(nrs->nBDF, nrs->nEXT);
    platform->o_mempool.slice0.copyFrom(mesh->o_LMM, mesh->Nlocal * sizeof(dfloat));
    mesh->o_LMM.free();
    mesh->o_LMM = platform->device.malloc(nrs->fieldOffset * nBDF, sizeof(dfloat));
    mesh->o_LMM.copyFrom(platform->o_mempool.slice0, mesh->Nlocal * sizeof(dfloat));
    platform->o_mempool.slice0.copyFrom(mesh->o_invLMM, mesh->Nlocal * sizeof(dfloat));
    mesh->o_invLMM.free();
    mesh->o_invLMM = platform->device.malloc(nrs->fieldOffset * nBDF, sizeof(dfloat));
    mesh->o_invLMM.copyFrom(platform->o_mempool.slice0, mesh->Nlocal * sizeof(dfloat));

    const int nAB = std::max(nrs->nEXT, mesh->nAB);
    mesh->U = (dfloat *)calloc(nrs->NVfields * nrs->fieldOffset * nAB, sizeof(dfloat));
    mesh->o_U = platform->device.malloc((nrs->NVfields * nAB * sizeof(dfloat)) * nrs->fieldOffset, mesh->U);
    mesh->o_Ue = platform->device.malloc((nrs->NVfields * nAB * sizeof(dfloat)) * nrs->fieldOffset, mesh->U);
    if (nrs->Nsubsteps)
      mesh->o_divU = platform->device.malloc(nrs->fieldOffset * nAB, sizeof(dfloat));
  }

  {
    const dlong Nstates = nrs->Nsubsteps ? std::max(nrs->nBDF, nrs->nEXT) : 1;
    if (nrs->Nsubsteps && platform->options.compareArgs("MOVING MESH", "TRUE"))
      nrs->o_relUrst =
          platform->device.malloc((Nstates * nrs->NVfields * sizeof(dfloat)) * nrs->cubatureOffset);
    else
      nrs->o_Urst = platform->device.malloc((Nstates * nrs->NVfields * sizeof(dfloat)) * nrs->cubatureOffset);
  }

  nrs->U =
      (dfloat *)calloc(nrs->NVfields * std::max(nrs->nBDF, nrs->nEXT) * nrs->fieldOffset, sizeof(dfloat));
  nrs->Ue = (dfloat *)calloc(nrs->NVfields * nrs->fieldOffset, sizeof(dfloat));
  nrs->P = (dfloat *)calloc(nrs->fieldOffset, sizeof(dfloat));
  nrs->BF = (dfloat *)calloc(nrs->NVfields * nrs->fieldOffset, sizeof(dfloat));
  nrs->FU = (dfloat *)calloc(nrs->NVfields * nrs->nEXT * nrs->fieldOffset, sizeof(dfloat));

  nrs->o_U = platform->device.malloc(nrs->NVfields * std::max(nrs->nBDF, nrs->nEXT) * nrs->fieldOffset *
                                         sizeof(dfloat),
                                     nrs->U);
  nrs->o_Ue = platform->device.malloc((nrs->NVfields * sizeof(dfloat)) * nrs->fieldOffset, nrs->Ue);
  nrs->o_P = platform->device.malloc(nrs->fieldOffset * sizeof(dfloat), nrs->P);
  nrs->o_BF = platform->device.malloc((nrs->NVfields * sizeof(dfloat)) * nrs->fieldOffset, nrs->BF);
  nrs->o_FU =
      platform->device.malloc((nrs->NVfields * nrs->nEXT * sizeof(dfloat)) * nrs->fieldOffset, nrs->FU);

  nrs->o_ellipticCoeff = device.malloc((2 * sizeof(dfloat)) * nrs->fieldOffset);

  int nProperties = 2;
  if (options.compareArgs("MESH SOLVER", "ELASTICITY"))
    nProperties = 4;
  nrs->prop = (dfloat *)calloc(nProperties * nrs->fieldOffset, sizeof(dfloat));
  for (int e = 0; e < mesh->Nelements; e++)
    for (int n = 0; n < mesh->Np; n++) {
      nrs->prop[0 * nrs->fieldOffset + e * mesh->Np + n] = mue;
      nrs->prop[1 * nrs->fieldOffset + e * mesh->Np + n] = rho;
    }

  nrs->o_prop = device.malloc((nProperties * sizeof(dfloat)) * nrs->fieldOffset, nrs->prop);
  nrs->o_mue = nrs->o_prop.slice((0 * sizeof(dfloat)) * nrs->fieldOffset);
  nrs->o_rho = nrs->o_prop.slice((1 * sizeof(dfloat)) * nrs->fieldOffset);
  if (options.compareArgs("MESH SOLVER", "ELASTICITY")) {
    nrs->o_meshMue = nrs->o_prop.slice((2 * sizeof(dfloat)) * nrs->fieldOffset);
    nrs->o_meshRho = nrs->o_prop.slice((3 * sizeof(dfloat)) * nrs->fieldOffset);
  }

  if (platform->options.compareArgs("CONSTANT FLOW RATE", "TRUE")) {
    nrs->o_Uc = platform->device.malloc((nrs->NVfields * sizeof(dfloat)) * nrs->fieldOffset);
    nrs->o_Pc = platform->device.malloc(nrs->fieldOffset * sizeof(dfloat));
    nrs->o_prevProp = device.malloc((2 * sizeof(dfloat)) * nrs->fieldOffset, nrs->prop);
  }

  nrs->div = (dfloat *)calloc(nrs->fieldOffset, sizeof(dfloat));
  nrs->o_div = device.malloc(nrs->fieldOffset * sizeof(dfloat), nrs->div);

  nrs->o_coeffEXT = platform->device.malloc(nrs->nEXT * sizeof(dfloat), nrs->coeffEXT);
  nrs->o_coeffBDF = platform->device.malloc(nrs->nBDF * sizeof(dfloat), nrs->coeffBDF);
  nrs->o_coeffSubEXT = platform->device.malloc(nrs->nEXT * sizeof(dfloat), nrs->coeffEXT);

  // meshParallelGatherScatterSetup(mesh, mesh->Nlocal, mesh->globalIds, platform->comm.mpiComm, OOGS_AUTO,
  // 0);
  nrs->gsh = oogs::setup(mesh->ogs, nrs->NVfields, nrs->fieldOffset, ogsDfloat, NULL, OOGS_AUTO);

  nrs->EToB = (int *)calloc(mesh->Nelements * mesh->Nfaces, sizeof(int));
  int cnt = 0;
  for (int e = 0; e < mesh->Nelements; e++) {
    for (int f = 0; f < mesh->Nfaces; f++) {
      nrs->EToB[cnt] = bcMap::id(mesh->EToB[f + e * mesh->Nfaces], "velocity");
      cnt++;
    }
  }
  nrs->o_EToB = device.malloc(mesh->Nelements * mesh->Nfaces * sizeof(int), nrs->EToB);

  if (platform->options.compareArgs("MESH SOLVER", "ELASTICITY")) {

    nrs->EToBMeshVelocity = (int *)calloc(mesh->Nelements * mesh->Nfaces, sizeof(int));
    int cnt = 0;
    for (int e = 0; e < mesh->Nelements; e++) {
      for (int f = 0; f < mesh->Nfaces; f++) {
        int bc = bcMap::id(mesh->EToB[f + e * mesh->Nfaces], "mesh");
        nrs->EToBMeshVelocity[cnt] = bcMap::id(mesh->EToB[f + e * mesh->Nfaces], "mesh");
        cnt++;
      }
    }
    nrs->o_EToBMeshVelocity =
        device.malloc(mesh->Nelements * mesh->Nfaces * sizeof(int), nrs->EToBMeshVelocity);
  }

  if (platform->options.compareArgs("VELOCITY REGULARIZATION METHOD", "RELAXATION")) {

    nrs->filterNc = -1;
    dfloat filterS;
    platform->options.getArgs("VELOCITY HPFRT STRENGTH", filterS);
    platform->options.getArgs("VELOCITY HPFRT MODES", nrs->filterNc);
    filterS = -1.0 * fabs(filterS);
    nrs->filterS = filterS;

    dfloat *A = filterSetup(nrs->meshV, nrs->filterNc);

    const dlong Nmodes = nrs->meshV->N + 1;

    nrs->o_filterMT = platform->device.malloc(Nmodes * Nmodes * sizeof(dfloat), A);

    free(A);
  }

  // build kernels
  std::string kernelName;
  const std::string suffix = "Hex3D";
  {
    const std::string section = "nrs-";
    kernelName = "nStagesSum3";
    nrs->nStagesSum3Kernel = platform->kernels.get(section + kernelName);

    kernelName = "computeFieldDotNormal";
    nrs->computeFieldDotNormalKernel = platform->kernels.get(section + kernelName);

    kernelName = "computeFaceCentroid";
    nrs->computeFaceCentroidKernel = platform->kernels.get(section + kernelName);

    {
      kernelName = "strongAdvectionVolume" + suffix;
      nrs->strongAdvectionVolumeKernel = platform->kernels.get(section + kernelName);
      kernelName = "strongAdvectionCubatureVolume" + suffix;
      nrs->strongAdvectionCubatureVolumeKernel = platform->kernels.get(section + kernelName);
    }

    kernelName = "curl" + suffix;
    nrs->curlKernel = platform->kernels.get(section + kernelName);

    kernelName = "SijOij" + suffix;
    nrs->SijOijKernel = platform->kernels.get(section + kernelName);

    kernelName = "SijOijMag2";
    nrs->SijOijMag2Kernel = platform->kernels.get(section + kernelName);

    kernelName = "gradientVolume" + suffix;
    nrs->gradientVolumeKernel = platform->kernels.get(section + kernelName);

    kernelName = "wGradientVolume" + suffix;
    nrs->wgradientVolumeKernel = platform->kernels.get(section + kernelName);

    {
      kernelName = "sumMakef";
      nrs->sumMakefKernel = platform->kernels.get(section + kernelName);
    }

    kernelName = "wDivergenceVolume" + suffix;
    nrs->wDivergenceVolumeKernel = platform->kernels.get(section + kernelName);
    kernelName = "divergenceVolume" + suffix;
    nrs->divergenceVolumeKernel = platform->kernels.get(section + kernelName);

    kernelName = "divergenceSurface" + suffix;
    nrs->divergenceSurfaceKernel = platform->kernels.get(section + kernelName);

    kernelName = "advectMeshVelocity" + suffix;
    nrs->advectMeshVelocityKernel = platform->kernels.get(section + kernelName);

    kernelName = "pressureRhs" + suffix;
    nrs->pressureRhsKernel = platform->kernels.get(section + kernelName);

    kernelName = "pressureStress" + suffix;
    nrs->pressureStressKernel = platform->kernels.get(section + kernelName);

    kernelName = "pressureDirichletBC" + suffix;
    nrs->pressureDirichletBCKernel = platform->kernels.get(section + kernelName);

    kernelName = "velocityRhs" + suffix;
    nrs->velocityRhsKernel = platform->kernels.get(section + kernelName);

    kernelName = "averageNormalBcType";
    nrs->averageNormalBcTypeKernel = platform->kernels.get(section + kernelName);

    kernelName = "fixZeroNormalMask";
    nrs->fixZeroNormalMaskKernel = platform->kernels.get(section + kernelName);

    kernelName = "applyZeroNormalMask";
    nrs->applyZeroNormalMaskKernel = platform->kernels.get(section + kernelName);

    kernelName = "initializeZeroNormalMask";
    nrs->initializeZeroNormalMaskKernel = platform->kernels.get(section + kernelName);

    kernelName = "velocityDirichletBC" + suffix;
    nrs->velocityDirichletBCKernel = platform->kernels.get(section + kernelName);

    kernelName = "velocityNeumannBC" + suffix;
    nrs->velocityNeumannBCKernel = platform->kernels.get(section + kernelName);

    kernelName = "UrstCubature" + suffix;
    nrs->UrstCubatureKernel = platform->kernels.get(section + kernelName);

    kernelName = "Urst" + suffix;
    nrs->UrstKernel = platform->kernels.get(section + kernelName);

    if (nrs->Nsubsteps) {
      if (platform->options.compareArgs("ADVECTION TYPE", "CUBATURE")) {
        kernelName = "subCycleStrongCubatureVolume" + suffix;
        nrs->subCycleStrongCubatureVolumeKernel = platform->kernels.get(section + kernelName);
      }
      kernelName = "subCycleStrongVolume" + suffix;
      nrs->subCycleStrongVolumeKernel = platform->kernels.get(section + kernelName);

      kernelName = "subCycleRKUpdate";
      nrs->subCycleRKUpdateKernel = platform->kernels.get(section + kernelName);
      kernelName = "subCycleRK";
      nrs->subCycleRKKernel = platform->kernels.get(section + kernelName);

      kernelName = "subCycleInitU0";
      nrs->subCycleInitU0Kernel = platform->kernels.get(section + kernelName);
    }

    kernelName = "extrapolate";
    nrs->extrapolateKernel = platform->kernels.get(section + kernelName);

    kernelName = "maskCopy";
    nrs->maskCopyKernel = platform->kernels.get(section + kernelName);

    kernelName = "maskCopy2";
    nrs->maskCopy2Kernel = platform->kernels.get(section + kernelName);

    kernelName = "mask";
    nrs->maskKernel = platform->kernels.get(section + kernelName);

    kernelName = "filterRT" + suffix;
    nrs->filterRTKernel = platform->kernels.get(section + kernelName);

    kernelName = "cfl" + suffix;
    nrs->cflKernel = platform->kernels.get(section + kernelName);

    kernelName = "pressureAddQtl";
    nrs->pressureAddQtlKernel = platform->kernels.get(section + kernelName);

    kernelName = "setEllipticCoeff";
    nrs->setEllipticCoeffKernel = platform->kernels.get(section + kernelName);
    kernelName = "setEllipticCoeffPressure";
    nrs->setEllipticCoeffPressureKernel = platform->kernels.get(section + kernelName);
  }

  if (nrs->Nscalar) {
    nrs->cds = cdsSetup(nrs, platform->options);
  }

  // get IC + t0 from nek
  double startTime;
  nek::copyFromNek(startTime);
  platform->options.setArgs("START TIME", to_string_f(startTime));

  if (platform->comm.mpiRank == 0)
    printf("calling udf_setup ... ");
  fflush(stdout);
  udf.setup(nrs);
  if (platform->comm.mpiRank == 0)
    printf("done\n");
  fflush(stdout);

  nrs->o_U.copyFrom(nrs->U);
  nrs->o_P.copyFrom(nrs->P);
  nrs->o_prop.copyFrom(nrs->prop);
  if (nrs->Nscalar) {
    nrs->cds->o_S.copyFrom(nrs->cds->S);
    nrs->cds->o_prop.copyFrom(nrs->cds->prop);
  }

  nrs->p0the = nrs->p0th[0];

  evaluateProperties(nrs, startTime);
  nrs->o_prop.copyTo(nrs->prop);
  if (nrs->Nscalar)
    nrs->cds->o_prop.copyTo(nrs->cds->prop);

  nek::ocopyToNek(startTime, 0);

  if (platform->comm.mpiRank == 0) std::cout << std::endl;
  printMeshMetrics(nrs->_mesh);
  
  printICMinMax(nrs);

  // setup elliptic solvers

  if (nrs->Nscalar) {
    cds_t *cds = nrs->cds;

    const int scalarWidth = getDigitsRepresentation(NSCALAR_MAX - 1);

    for (int is = 0; is < cds->NSfields; is++) {
      std::stringstream ss;
      ss << std::setfill('0') << std::setw(scalarWidth) << is;
      std::string sid = ss.str();

      if (!cds->compute[is])
        continue;

      mesh_t *mesh;
      (is) ? mesh = cds->meshV : mesh = cds->mesh[0]; // only first scalar can be a CHT mesh

      if (platform->comm.mpiRank == 0)
        std::cout << "================= ELLIPTIC SETUP SCALAR" << sid << " ===============\n";

      int nbrBIDs = bcMap::size(0);
      if (nrs->cht && is == 0)
        nbrBIDs = bcMap::size(1);
      for (int bID = 1; bID <= nbrBIDs; bID++) {
        std::string bcTypeText(bcMap::text(bID, "scalar" + sid));
        if (platform->comm.mpiRank == 0)
          printf("bID %d -> bcType %s\n", bID, bcTypeText.c_str());
      }

      cds->solver[is] = new elliptic_t();
      cds->solver[is]->name = "scalar" + sid;
      cds->solver[is]->blockSolver = 0;
      cds->solver[is]->Nfields = 1;
      cds->solver[is]->fieldOffset = nrs->fieldOffset;
      cds->solver[is]->o_wrk = o_mempoolElliptic;
      cds->solver[is]->mesh = mesh;
      cds->solver[is]->dim = cds->dim;
      cds->solver[is]->elementType = cds->elementType;

      const int coeffField = platform->options.compareArgs("SCALAR" + sid + " ELLIPTIC COEFF FIELD", "TRUE");
      cds->solver[is]->coeffField = coeffField;
      cds->solver[is]->coeffFieldPreco = coeffField;
      cds->solver[is]->poisson = 0;

      platform->linAlg->fill(2 * nrs->fieldOffset, 1.0, nrs->o_ellipticCoeff);
      cds->solver[is]->o_lambda = cds->o_ellipticCoeff;
      cds->solver[is]->loffset = 0; // not used

      cds->solver[is]->EToB = (int *)calloc(mesh->Nelements * mesh->Nfaces, sizeof(int));
      for (dlong e = 0; e < mesh->Nelements; e++) {
        for (int f = 0; f < mesh->Nfaces; f++) {
          const int bID = mesh->EToB[f + e * mesh->Nfaces];
          cds->solver[is]->EToB[f + e * mesh->Nfaces] = bcMap::ellipticType(bID, "scalar" + sid);
        }
      }

      ellipticSolveSetup(cds->solver[is]);
    }
  }

  if (nrs->flow) {

    if (platform->comm.mpiRank == 0)
      printf("================ ELLIPTIC SETUP VELOCITY ================\n");

    nrs->uvwSolver = NULL;

    bool unalignedBoundary = bcMap::unalignedRobinBoundary("velocity");
    if (unalignedBoundary) {
      if (!options.compareArgs("VELOCITY BLOCK SOLVER", "TRUE")) {
        if (platform->comm.mpiRank == 0)
          printf("ERROR: SHL or unaligned SYM boundaries require solver = pcg+block\n");
        ABORT(EXIT_FAILURE);
      }
    }

    if (platform->options.compareArgs("VELOCITY BLOCK SOLVER", "TRUE"))
      nrs->uvwSolver = new elliptic_t();

    for (int bID = 1; bID <= bcMap::size(0); bID++) {
      std::string bcTypeText(bcMap::text(bID, "velocity"));
      if (platform->comm.mpiRank == 0)
        printf("bID %d -> bcType %s\n", bID, bcTypeText.c_str());
    }

    // coeff used by ellipticSetup to detect allNeumann
    platform->linAlg->fill(2 * nrs->fieldOffset, 1.0, nrs->o_ellipticCoeff);

    const int velCoeffField = platform->options.compareArgs("VELOCITY ELLIPTIC COEFF FIELD", "TRUE");

    if (nrs->uvwSolver) {
      nrs->uvwSolver->name = "velocity";
      nrs->uvwSolver->blockSolver = 1;
      nrs->uvwSolver->stressForm = 0;
      if (options.compareArgs("VELOCITY STRESSFORMULATION", "TRUE"))
        nrs->uvwSolver->stressForm = 1;
      nrs->uvwSolver->Nfields = nrs->NVfields;
      nrs->uvwSolver->fieldOffset = nrs->fieldOffset;
      nrs->uvwSolver->o_wrk = o_mempoolElliptic;
      nrs->uvwSolver->mesh = mesh;
      nrs->uvwSolver->dim = nrs->dim;
      nrs->uvwSolver->elementType = nrs->elementType;
      nrs->uvwSolver->coeffField = velCoeffField;
      nrs->uvwSolver->coeffFieldPreco = velCoeffField;
      nrs->uvwSolver->o_lambda = nrs->o_ellipticCoeff;
      nrs->uvwSolver->loffset = 0; // use same ellipticCoeff for u,v and w
      nrs->uvwSolver->poisson = 0;
      nrs->uvwSolver->EToB =
          (int *)calloc(mesh->Nelements * mesh->Nfaces * nrs->uvwSolver->Nfields, sizeof(int));
      for (int fld = 0; fld < nrs->uvwSolver->Nfields; fld++) {
        std::string key;
        if (fld == 0)
          key = "x-velocity";
        if (fld == 1)
          key = "y-velocity";
        if (fld == 2)
          key = "z-velocity";
        for (dlong e = 0; e < mesh->Nelements; e++) {
          for (int f = 0; f < mesh->Nfaces; f++) {
            const int offset = fld * mesh->Nelements * mesh->Nfaces;
            const int bID = mesh->EToB[f + e * mesh->Nfaces];
            nrs->uvwSolver->EToB[f + e * mesh->Nfaces + offset] = bcMap::ellipticType(bID, key);
          }
        }
      }

      if (unalignedBoundary) {
        nrs->o_zeroNormalMaskVelocity = 
          platform->device.malloc((nrs->uvwSolver->Nfields * sizeof(dfloat)) * 
                                  nrs->uvwSolver->fieldOffset);
        nrs->o_EToBVVelocity = platform->device.malloc(nrs->meshV->Nlocal * sizeof(int));
        createEToBV(nrs->meshV, nrs->uvwSolver->EToB, nrs->o_EToBVVelocity);
        auto o_EToB = 
          platform->device.malloc(mesh->Nelements * mesh->Nfaces * nrs->uvwSolver->Nfields * 
                                  sizeof(int), nrs->uvwSolver->EToB);
        createZeroNormalMask(nrs, o_EToB, nrs->o_EToBVVelocity, nrs->o_zeroNormalMaskVelocity);

        nrs->uvwSolver->applyZeroNormalMask =
            [nrs](dlong Nelements, occa::memory &o_elementList, occa::memory &o_x) {
              applyZeroNormalMask(nrs,
                                  Nelements,
                                  o_elementList,
                                  nrs->uvwSolver->o_EToB,
                                  nrs->o_zeroNormalMaskVelocity,
                                  o_x);
            };
      }
      ellipticSolveSetup(nrs->uvwSolver);
    }
    else {
      nrs->uSolver = new elliptic_t();
      nrs->uSolver->name = "velocity";
      nrs->uSolver->blockSolver = 0;
      nrs->uSolver->Nfields = 1;
      nrs->uSolver->fieldOffset = nrs->fieldOffset;
      nrs->uSolver->o_wrk = o_mempoolElliptic;
      nrs->uSolver->mesh = mesh;
      nrs->uSolver->dim = nrs->dim;
      nrs->uSolver->elementType = nrs->elementType;
      nrs->uSolver->coeffField = velCoeffField;
      nrs->uSolver->coeffFieldPreco = velCoeffField;
      nrs->uSolver->o_lambda = nrs->o_ellipticCoeff;
      nrs->uSolver->loffset = 0; // not used
      nrs->uSolver->poisson = 0;
      nrs->uSolver->EToB = (int *)calloc(mesh->Nelements * mesh->Nfaces, sizeof(int));
      for (dlong e = 0; e < mesh->Nelements; e++) {
        for (int f = 0; f < mesh->Nfaces; f++) {
          const int bID = mesh->EToB[f + e * mesh->Nfaces];
          nrs->uSolver->EToB[f + e * mesh->Nfaces] = bcMap::ellipticType(bID, "x-velocity");
        }
      }

      ellipticSolveSetup(nrs->uSolver);

      nrs->vSolver = new elliptic_t();
      nrs->vSolver->name = "velocity";
      nrs->vSolver->blockSolver = 0;
      nrs->vSolver->Nfields = 1;
      nrs->vSolver->fieldOffset = nrs->fieldOffset;
      nrs->vSolver->o_wrk = o_mempoolElliptic;
      nrs->vSolver->mesh = mesh;
      nrs->vSolver->dim = nrs->dim;
      nrs->vSolver->elementType = nrs->elementType;
      nrs->vSolver->coeffField = velCoeffField;
      nrs->vSolver->coeffFieldPreco = velCoeffField;
      nrs->vSolver->o_lambda = nrs->o_ellipticCoeff;
      nrs->vSolver->loffset = 0; // not used 
      nrs->vSolver->poisson = 0;
      nrs->vSolver->EToB = (int *)calloc(mesh->Nelements * mesh->Nfaces, sizeof(int));
      for (dlong e = 0; e < mesh->Nelements; e++) {
        for (int f = 0; f < mesh->Nfaces; f++) {
          const int bID = mesh->EToB[f + e * mesh->Nfaces];
          nrs->vSolver->EToB[f + e * mesh->Nfaces] = bcMap::ellipticType(bID, "y-velocity");
        }
      }

      ellipticSolveSetup(nrs->vSolver);

      nrs->wSolver = new elliptic_t();
      nrs->wSolver->name = "velocity";
      nrs->wSolver->blockSolver = 0;
      nrs->wSolver->Nfields = 1;
      nrs->wSolver->fieldOffset = nrs->fieldOffset;
      nrs->wSolver->o_wrk = o_mempoolElliptic;
      nrs->wSolver->mesh = mesh;
      nrs->wSolver->dim = nrs->dim;
      nrs->wSolver->elementType = nrs->elementType;
      nrs->wSolver->coeffField = velCoeffField;
      nrs->wSolver->coeffFieldPreco = velCoeffField;
      nrs->wSolver->o_lambda = nrs->o_ellipticCoeff;
      nrs->wSolver->loffset = 0; // not used
      nrs->wSolver->poisson = 0;
      nrs->wSolver->EToB = (int *)calloc(mesh->Nelements * mesh->Nfaces, sizeof(int));
      for (dlong e = 0; e < mesh->Nelements; e++) {
        for (int f = 0; f < mesh->Nfaces; f++) {
          const int bID = mesh->EToB[f + e * mesh->Nfaces];
          nrs->wSolver->EToB[f + e * mesh->Nfaces] = bcMap::ellipticType(bID, "z-velocity");
        }
      }

      ellipticSolveSetup(nrs->wSolver);
    }
  } // flow

  if (nrs->flow) {
    if (platform->comm.mpiRank == 0)
      printf("================ ELLIPTIC SETUP PRESSURE ================\n");

    nrs->pSolver = new elliptic_t();
    nrs->pSolver->name = "pressure";
    nrs->pSolver->blockSolver = 0;
    nrs->pSolver->Nfields = 1;
    nrs->pSolver->fieldOffset = nrs->fieldOffset;
    nrs->pSolver->o_wrk = o_mempoolElliptic;
    nrs->pSolver->mesh = mesh;
    nrs->pSolver->dim = nrs->dim;
    nrs->pSolver->elementType = nrs->elementType;

    nrs->pSolver->coeffField = platform->options.compareArgs("PRESSURE ELLIPTIC COEFF FIELD", "TRUE");
    nrs->pSolver->coeffFieldPreco = platform->options.compareArgs("PRESSURE PRECONDITIONER COEFF FIELD", "TRUE"); 
    nrs->pSolver->poisson = 1;

    // lambda0 = 1/rho  lambda1 = 0
    platform->linAlg->fill(2 * nrs->fieldOffset, 0.0, nrs->o_ellipticCoeff);
    nrs->o_ellipticCoeff.copyFrom(nrs->o_rho, nrs->fieldOffset * sizeof(dfloat));
    platform->linAlg->ady(mesh->Nlocal, 1.0, nrs->o_ellipticCoeff);
    nrs->pSolver->o_lambda = nrs->o_ellipticCoeff;
    nrs->pSolver->loffset = 0; // not used 
    {
      const std::vector<int> levels = determineMGLevels("pressure");
      nrs->pSolver->nLevels = levels.size();
      nrs->pSolver->levels = (int *)calloc(nrs->pSolver->nLevels, sizeof(int));
      for (int i = 0; i < nrs->pSolver->nLevels; ++i)
        nrs->pSolver->levels[i] = levels.at(i);
    }
    nrs->pSolver->EToB = (int *)calloc(mesh->Nelements * mesh->Nfaces, sizeof(int));
    for (dlong e = 0; e < mesh->Nelements; e++) {
      for (int f = 0; f < mesh->Nfaces; f++) {
        const int bID = mesh->EToB[f + e * mesh->Nfaces];
        nrs->pSolver->EToB[f + e * mesh->Nfaces] = bcMap::ellipticType(bID, "pressure");
      }
    }

    ellipticSolveSetup(nrs->pSolver);

  } // flow
  if (nrs->flow) {

    if (options.compareArgs("MESH SOLVER", "ELASTICITY")) {

      bool unalignedBoundary = bcMap::unalignedRobinBoundary("mesh");
      if (unalignedBoundary) {
        if (platform->comm.mpiRank == 0) {
          printf("ERROR: unaligned SYM/SHL boundary condition are currently not supported with the mesh "
                 "solver.\n");
        }
        ABORT(EXIT_FAILURE);
      }

      if (platform->comm.mpiRank == 0)
        printf("================ ELLIPTIC SETUP MESH ================\n");

      for (int bID = 1; bID <= bcMap::size(0); bID++) {
        std::string bcTypeText(bcMap::text(bID, "mesh"));
        if (platform->comm.mpiRank == 0)
          printf("bID %d -> bcType %s\n", bID, bcTypeText.c_str());
      }

      const int meshCoeffField = platform->options.compareArgs("MESH ELLIPTIC COEFF FIELD", "TRUE");
      platform->linAlg->fill(2 * nrs->fieldOffset, 1.0, nrs->o_ellipticCoeff);

      nrs->meshSolver = new elliptic_t();
      nrs->meshSolver->name = "mesh";
      nrs->meshSolver->blockSolver = 1;
      nrs->meshSolver->stressForm = 0;
      if (options.compareArgs("MESH STRESSFORMULATION", "TRUE"))
        nrs->meshSolver->stressForm = 1;
      nrs->meshSolver->Nfields = nrs->NVfields;
      nrs->meshSolver->fieldOffset = nrs->fieldOffset;
      nrs->meshSolver->o_wrk = o_mempoolElliptic;
      nrs->meshSolver->mesh = mesh;
      nrs->meshSolver->dim = nrs->dim;
      nrs->meshSolver->elementType = nrs->elementType;
      nrs->meshSolver->coeffField = meshCoeffField;
      nrs->meshSolver->coeffFieldPreco = meshCoeffField;
      nrs->meshSolver->o_lambda = nrs->o_ellipticCoeff;
      nrs->meshSolver->loffset = 0; // use same ellipticCoeff for u,v and w
      nrs->meshSolver->poisson = 0;

      nrs->meshSolver->EToB =
          (int *)calloc(mesh->Nelements * mesh->Nfaces * nrs->meshSolver->Nfields, sizeof(int));
      for (int fld = 0; fld < nrs->meshSolver->Nfields; fld++) {
        std::string key;
        if (fld == 0)
          key = "x-mesh";
        if (fld == 1)
          key = "y-mesh";
        if (fld == 2)
          key = "z-mesh";
        for (dlong e = 0; e < mesh->Nelements; e++) {
          for (int f = 0; f < mesh->Nfaces; f++) {
            const int offset = fld * mesh->Nelements * mesh->Nfaces;
            const int bID = mesh->EToB[f + e * mesh->Nfaces];
            nrs->meshSolver->EToB[f + e * mesh->Nfaces + offset] = bcMap::ellipticType(bID, key);
          }
        }
      }

      if (unalignedBoundary) {
        nrs->o_zeroNormalMaskMeshVelocity = 
          platform->device.malloc((nrs->meshSolver->Nfields * sizeof(dfloat)) * 
                                  nrs->meshSolver->fieldOffset);
        nrs->o_EToBVMeshVelocity = platform->device.malloc(mesh->Nlocal * sizeof(int));
        auto o_EToB = platform->device.malloc(mesh->Nelements * mesh->Nfaces * nrs->meshSolver->Nfields, 
                                              sizeof(int), nrs->meshSolver->EToB);
        createEToBV(nrs->meshV, nrs->meshSolver->EToB, nrs->o_EToBVMeshVelocity);
        createZeroNormalMask(nrs, o_EToB, nrs->o_EToBVMeshVelocity, nrs->o_zeroNormalMaskMeshVelocity);
        nrs->meshSolver->applyZeroNormalMask =
            [nrs](dlong Nelements, occa::memory &o_elementList, occa::memory &o_x) {
              applyZeroNormalMask(nrs,
                                  Nelements,
                                  o_elementList,
                                  nrs->meshSolver->o_EToB,
                                  nrs->o_zeroNormalMaskMeshVelocity,
                                  o_x);
            };
      }
      ellipticSolveSetup(nrs->meshSolver);
    }
  }
}
