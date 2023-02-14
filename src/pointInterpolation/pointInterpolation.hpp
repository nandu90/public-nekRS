#if !defined(nekrs_interp_hpp_)
#define nekrs_interp_hpp_

#include <vector>
#include <memory>
#include "nrssys.hpp"
#include "findpts.hpp"

class nrs_t;

using findpts::TimerLevel;

class pointInterpolation_t {
public:
  pointInterpolation_t(nrs_t *nrs_, dfloat newton_tol_ = 0, bool mySession_ = true);
  ~pointInterpolation_t() = default;

  // Finds the process, element, and reference coordinates of the given points
  void find(bool printWarnings = true);

  void
  eval(dlong nFields, dlong inputFieldOffset, occa::memory o_in, dlong outputFieldOffset, occa::memory o_out);

  auto *ptr() { return findpts_.get(); }
  auto &data() {return data_;}

  // add points on device
  void addPoints(int n, occa::memory o_x, occa::memory o_y, occa::memory o_z);

  // add points on host
  void addPoints(int n, dfloat *x, dfloat *y, dfloat *z);

  // set timer level
  void setTimerLevel(TimerLevel level);
  TimerLevel getTimerLevel() const;

  // set timer name
  // this is used to prefix the timer names
  void setTimerName(std::string name);

private:
  nrs_t *nrs;
  double newton_tol;
  std::string timerName = "";
  TimerLevel timerLevel = TimerLevel::None;
  std::unique_ptr<findpts::findpts_t> findpts_;
  findpts::data_t data_;
  bool mySession;

  bool pointsAdded = false;

  // correponds  to which addPoints overload is called
  bool useHostPoints = false;
  bool useDevicePoints = false;

  int nPoints;

  dfloat * _x;
  dfloat * _y;
  dfloat * _z;

  occa::memory _o_x;
  occa::memory _o_y;
  occa::memory _o_z;

  // for storing host points to output when a particle leaves the domain
  std::vector<dfloat> h_x_vec;
  std::vector<dfloat> h_y_vec;
  std::vector<dfloat> h_z_vec;
};

#endif