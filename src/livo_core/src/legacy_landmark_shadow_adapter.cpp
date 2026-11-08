#include "legacy_landmark_shadow_adapter.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

namespace livo
{
namespace
{

bool finiteSample(const LegacyPlaneLandmarkSample & sample)
{
  const double u_norm = sample.identity_u_axis_world.norm();
  const double v_norm = sample.identity_v_axis_world.norm();
  return
    sample.plane.valid() &&
    sample.geometry_revision != 0U &&
    sample.camera_id < 4U &&
    sample.identity_origin_world_m.allFinite() &&
    sample.identity_u_axis_world.allFinite() &&
    sample.identity_v_axis_world.allFinite() &&
    sample.position_world_m.allFinite() &&
    sample.position_covariance_m2.allFinite() &&
    sample.normal_world.allFinite() &&
    sample.normal_world.norm() > 1.0e-9 &&
    sample.plane_covariance.allFinite() &&
    std::isfinite(u_norm) &&
    std::isfinite(v_norm) &&
    std::abs(u_norm - 1.0) <= 1.0e-6 &&
    std::abs(v_norm - 1.0) <= 1.0e-6 &&
    std::abs(
      sample.identity_u_axis_world.dot(
        sample.identity_v_axis_world)) <= 1.0e-6;
}

bool finiteCellCoordinate(
  const double coordinate, std::int32_t & result)
{
  if (!std::isfinite(coordinate) ||
      coordinate <
        static_cast<double>(
          std::numeric_limits<std::int32_t>::min()) ||
      coordinate >
        static_cast<double>(
          std::numeric_limits<std::int32_t>::max()))
  {
    return false;
  }
  result = static_cast<std::int32_t>(coordinate);
  return true;
}

bool seedFromSample(
  const LegacyPlaneLandmarkSample & sample,
  const double cell_size_m,
  GeometrySeed & seed)
{
  if (!resolveLegacyPlaneLandmarkIdentity(
      sample, cell_size_m, seed.support_key,
      seed.geometry_version))
  {
    return false;
  }
  seed.active_plane_lease =
    sample.active_plane_lease;
  seed.position_world_m = sample.position_world_m;
  seed.position_covariance_m2 =
    sample.position_covariance_m2;
  seed.normal_world = sample.normal_world.normalized();
  seed.plane_covariance = sample.plane_covariance;
  seed.local_basis.origin_world_m =
    sample.identity_origin_world_m;
  seed.local_basis.u_axis_world =
    sample.identity_u_axis_world;
  seed.local_basis.v_axis_world =
    sample.identity_v_axis_world;
  seed.camera_id = sample.camera_id;
  return true;
}

const LandmarkSnapshot * findBySupportKey(
  const ReadSnapshot & snapshot,
  const GeometrySupportKey & support_key)
{
  const auto landmark = std::find_if(
    snapshot.landmarks.begin(), snapshot.landmarks.end(),
    [&support_key](const LandmarkSnapshot & candidate) {
      return candidate.support_key == support_key;
    });
  return
    landmark == snapshot.landmarks.end() ?
    nullptr : &*landmark;
}

}  // namespace

bool resolveLegacyPlaneLandmarkIdentity(
  const LegacyPlaneLandmarkSample & sample,
  const double plane_cell_size_m,
  GeometrySupportKey & support_key,
  GeometryVersion & geometry_version)
{
  if (!finiteSample(sample) ||
      !std::isfinite(plane_cell_size_m) ||
      plane_cell_size_m <= 0.0)
  {
    return false;
  }
  const Eigen::Vector3d offset =
    sample.position_world_m -
    sample.identity_origin_world_m;
  std::int32_t cell_u = 0;
  std::int32_t cell_v = 0;
  if (!finiteCellCoordinate(
      std::floor(
        offset.dot(sample.identity_u_axis_world) /
        plane_cell_size_m),
      cell_u) ||
      !finiteCellCoordinate(
      std::floor(
        offset.dot(sample.identity_v_axis_world) /
        plane_cell_size_m),
      cell_v))
  {
    return false;
  }

  support_key.plane = sample.plane;
  support_key.cell =
    PlaneLocalCell{cell_u, cell_v, sample.level};
  geometry_version =
    GeometryVersion{sample.plane, sample.geometry_revision};
  return true;
}

LegacyLandmarkShadowAdapter::LegacyLandmarkShadowAdapter(
  LegacyLandmarkShadowConfig config)
: config_(config)
{
}

bool LegacyLandmarkShadowAdapter::configValid() const
{
  return
    std::isfinite(config_.plane_cell_size_m) &&
    config_.plane_cell_size_m > 0.0 &&
    config_.maximum_seeds_per_batch > 0U;
}

LandmarkShadowBootstrapResult
LegacyLandmarkShadowAdapter::bootstrap(
  MultiViewLandmarkStore & store,
  const LandmarkShadowBootstrapRequest & request) const
{
  LandmarkShadowBootstrapResult result;
  result.input_sample_count = request.samples.size();
  if (!config_.enabled)
  {
    result.status = LandmarkShadowBootstrapStatus::kDisabled;
    return result;
  }
  if (!configValid())
  {
    result.status =
      LandmarkShadowBootstrapStatus::kInvalidConfiguration;
    return result;
  }
  if (
    request.batch.value == 0U ||
    request.geometry.value == 0U ||
    !request.active_planes.valid() ||
    !request.lidar_plane_stable)
  {
    result.status =
      LandmarkShadowBootstrapStatus::kProofRejected;
    return result;
  }
  const LidarAnchoredCommitToken token(
    request.geometry, request.expected_revisions);

  std::map<GeometrySupportKey, GeometrySeed> unique_seeds;
  for (const LegacyPlaneLandmarkSample & sample :
       request.samples)
  {
    GeometrySeed seed;
    if (!seedFromSample(
        sample, config_.plane_cell_size_m, seed))
    {
      continue;
    }
    const auto existing =
      unique_seeds.find(seed.support_key);
    if (existing == unique_seeds.end() ||
        seed.geometry_version.revision >
          existing->second.geometry_version.revision ||
        (seed.geometry_version.revision ==
           existing->second.geometry_version.revision &&
         seed.position_covariance_m2.trace() <
           existing->second.position_covariance_m2.trace()))
    {
      unique_seeds[seed.support_key] = std::move(seed);
    }
  }
  result.unique_seed_count = unique_seeds.size();
  if (unique_seeds.empty())
  {
    result.status =
      LandmarkShadowBootstrapStatus::kNoValidSeed;
    return result;
  }

  const ReadSnapshot snapshot =
    store.snapshot(
      request.geometry,
      request.expected_revisions,
      request.active_planes);
  LandmarkQuery query;
  query.batch = request.batch;
  query.geometry = request.geometry;
  query.active_planes = request.active_planes;
  query.expected_revisions =
    request.expected_revisions;
  query.geometry_seeds.reserve(std::min(
    unique_seeds.size(),
    config_.maximum_seeds_per_batch));
  for (const auto & entry : unique_seeds)
  {
    const LandmarkSnapshot * existing =
      findBySupportKey(snapshot, entry.first);
    if (existing != nullptr)
    {
      continue;
    }
    query.geometry_seeds.push_back(entry.second);
    if (query.geometry_seeds.size() >=
        config_.maximum_seeds_per_batch)
    {
      break;
    }
  }
  result.staged_seed_count = query.geometry_seeds.size();
  if (query.geometry_seeds.empty())
  {
    result.status =
      LandmarkShadowBootstrapStatus::kNoGeometryChange;
    result.commit_token = token;
    return result;
  }

  PreparedLandmarkBatch prepared =
    store.prepare(snapshot, query);
  if (!prepared.valid())
  {
    result.status =
      LandmarkShadowBootstrapStatus::kPrepareRejected;
    return result;
  }

  const MapCommitProof proof = BootstrapGate::mint(
    token, prepared);

  result.commit =
    store.commit(prepared.take_delta(), proof);
  result.status =
    result.commit.status == CommitStatus::kCommitted ?
    LandmarkShadowBootstrapStatus::kCommitted :
    LandmarkShadowBootstrapStatus::kCommitRejected;
  if (result.status == LandmarkShadowBootstrapStatus::kCommitted)
  {
    result.commit_token = token;
  }
  return result;
}

const char * landmarkShadowBootstrapStatusName(
  const LandmarkShadowBootstrapStatus status)
{
  switch (status)
  {
    case LandmarkShadowBootstrapStatus::kDisabled:
      return "disabled";
    case LandmarkShadowBootstrapStatus::kInvalidConfiguration:
      return "invalid_configuration";
    case LandmarkShadowBootstrapStatus::kNoValidSeed:
      return "no_valid_seed";
    case LandmarkShadowBootstrapStatus::kNoGeometryChange:
      return "no_geometry_change";
    case LandmarkShadowBootstrapStatus::kPrepareRejected:
      return "prepare_rejected";
    case LandmarkShadowBootstrapStatus::kProofRejected:
      return "proof_rejected";
    case LandmarkShadowBootstrapStatus::kCommitted:
      return "committed";
    case LandmarkShadowBootstrapStatus::kCommitRejected:
      return "commit_rejected";
  }
  return "unknown";
}

}  // namespace livo
