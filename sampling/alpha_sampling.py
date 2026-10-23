"""Texture-aware surface sampling for schema v2 (static, mip-zero alpha masks)."""
from pathlib import Path
import math
import numpy as np


class SurfaceData:
    def __init__(self, folder, scene, vertices, triangles):
        self.vertices, self.triangles = vertices, triangles
        self.uv = np.fromfile(Path(folder) / 'uvs.f32', dtype='<f4').reshape(-1, 2)
        if len(self.uv) != len(vertices) or not np.isfinite(self.uv).all():
            raise ValueError('Invalid UV0 buffer')
        self.materials = scene['materials']
        self.surfaces = scene['surfaces']
        self.face_surface = np.full(len(triangles), -1, np.int32)
        self.face_material = np.full(len(triangles), -1, np.int32)
        for i, surface in enumerate(self.surfaces):
            start, size = surface['triangleStart'], surface['triangleCount']
            material = surface['materialIndex']
            if start < 0 or size <= 0 or start + size > len(triangles) or not 0 <= material < len(self.materials):
                raise ValueError('Invalid surface triangle span/material index')
            if np.any(self.face_surface[start:start + size] != -1):
                raise ValueError('Overlapping surface spans')
            self.face_surface[start:start + size] = i
            self.face_material[start:start + size] = material
        if np.any(self.face_surface < 0):
            raise ValueError('Surface spans do not cover every triangle')
        self.textures = {}
        for i, mat in enumerate(self.materials):
            if mat['alphaMode'] not in ('opaque', 'clip'):
                raise ValueError('Unsupported alpha mode')
            if mat['alphaMode'] == 'clip' and mat.get('alphaMaskFile'):
                path = (Path(folder) / mat['alphaMaskFile']).resolve()
                if not path.is_relative_to(Path(folder).resolve()):
                    raise ValueError('Alpha mask outside export folder')
                a = np.fromfile(path, dtype=np.uint8)
                if mat['width'] <= 0 or mat['height'] <= 0 or a.size != mat['width'] * mat['height']:
                    raise ValueError('Invalid alpha mask dimensions')
                self.textures[i] = a.reshape(mat['height'], mat['width']).astype(np.float32) / 255
        corners = vertices[triangles, :3]
        areas = .5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
        self.raw_area = float(areas.sum())
        if not math.isfinite(self.raw_area) or self.raw_area <= 0:
            raise ValueError('No nondegenerate surface area')
        self.probabilities = areas / self.raw_area

    def candidates(self, count, rng):
        faces = rng.choice(len(self.triangles), count, p=self.probabilities)
        root, v = np.sqrt(rng.random(count)), rng.random(count)
        bary = np.column_stack((1-root, root*(1-v), root*v))
        uv = np.einsum('ni,nij->nj', bary, self.uv[self.triangles[faces]])
        ids = self.face_material[faces]
        keep = np.ones(count, bool)
        for mid in np.unique(ids):
            mat = self.materials[mid]
            if mat['alphaMode'] != 'clip':
                continue
            rows = np.flatnonzero(ids == mid)
            alpha = np.ones(len(rows))
            if mid in self.textures:
                scale, offset = mat['uvScale'], mat['uvOffset']
                q = uv[rows] * [scale['x'], scale['y']] + [offset['x'], offset['y']]
                alpha = texture_alpha(self.textures[mid], q, mat['wrapU'], mat['wrapV'], mat['filter'])
            keep[rows] = alpha * mat.get('alphaMultiplier', 1) >= mat['cutoff']
        return faces, bary, keep


def address(index, size, mode):
    if mode == 'Repeat':
        return index % size
    if mode == 'Mirror':
        i = index % (2*size)
        return np.where(i < size, i, 2*size-1-i)
    if mode == 'MirrorOnce':
        return np.clip(np.where(index < 0, -index-1, index), 0, size-1)
    if mode == 'Clamp':
        return np.clip(index, 0, size-1)
    raise ValueError('Unsupported texture wrap mode: ' + mode)


def texture_alpha(alpha, uv, wrap_u='Repeat', wrap_v='Repeat', filtering='bilinear'):
    """Unity UV origin bottom-left; exported mask bottom row first, no vertical flip."""
    height, width = alpha.shape
    if filtering == 'point':
        return alpha[address(np.floor(uv[:, 1]*height).astype(np.int64), height, wrap_v),
                     address(np.floor(uv[:, 0]*width).astype(np.int64), width, wrap_u)]
    if filtering != 'bilinear':
        raise ValueError('Unsupported alpha filter')
    q = uv * [width, height] - .5
    xy = np.floor(q).astype(np.int64)
    t = q - xy
    x0, x1 = address(xy[:, 0], width, wrap_u), address(xy[:, 0]+1, width, wrap_u)
    y0, y1 = address(xy[:, 1], height, wrap_v), address(xy[:, 1]+1, height, wrap_v)
    return ((1-t[:, 1])*((1-t[:, 0])*alpha[y0, x0] + t[:, 0]*alpha[y0, x1]) +
            t[:, 1]*((1-t[:, 0])*alpha[y1, x0] + t[:, 0]*alpha[y1, x1]))


def sample_textured(folder, scene, vertices, triangles, count, method, seed):
    surface = SurfaceData(folder, scene, vertices, triangles)
    # Independent fixed-size pilot estimates alpha-visible physical area. Do not
    # infer area from the acceptance fraction of a stopping-time rejection loop.
    pilot_rng, rng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
    pilot_count = 100000
    pf, _, pk = surface.candidates(pilot_count, pilot_rng)
    fraction = float(pk.mean())
    if fraction == 0:
        raise ValueError('No alpha-visible area in pilot; inspect masks and UVs')
    area = surface.raw_area * fraction
    needed = count * (5 if method == 'poisson' else 1)
    faces_list, bary_list, have, attempted, rejected = [], [], 0, 0, 0
    while have < needed:
        batch = min(100000, max(1024, math.ceil((needed-have) / fraction * 1.1)))
        f, b, k = surface.candidates(batch, rng)
        attempted += batch
        rejected += int((~k).sum())
        take = min(int(k.sum()), needed-have)
        faces_list.append(f[k][:take]); bary_list.append(b[k][:take]); have += take
        if attempted > max(1000000, needed * 1000):
            raise ValueError('Alpha rejection budget exceeded; inspect masks')
    faces, bary = np.concatenate(faces_list), np.concatenate(bary_list)
    points = np.einsum('ni,nij->nj', bary, vertices[triangles[faces], :3])
    normals = np.einsum('ni,nij->nj', bary, vertices[triangles[faces], 3:])
    norm = np.linalg.norm(normals, axis=1)
    if np.any(norm < 1e-8):
        raise ValueError('Interpolated normals cancel')
    normals /= norm[:, None]
    if method == 'poisson':
        import open3d as o3d
        from scipy.spatial import cKDTree
        o3d.utility.random.seed(seed)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.normals = o3d.utility.Vector3dVector(normals)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices[:, :3])
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        # Elimination only: never regenerate samples on full transparent cards.
        selected = mesh.sample_points_poisson_disk(count, pcl=cloud, use_triangle_normal=False)
        distance, indices = cKDTree(points).query(np.asarray(selected.points))
        if np.any(distance > 1e-9):
            raise RuntimeError('Poisson output is not a subset of alpha-valid candidates')
        points, normals, faces = points[indices], normals[indices], faces[indices]
    mid = surface.face_material[faces]
    categories = np.array([m.get('category', 0) for m in surface.materials], np.int8)[mid]
    doubles = np.array([m['doubleSided'] for m in surface.materials], bool)[mid]
    # Pair only AFTER elimination: coincident opposite faces must not eliminate
    # each other. Each side carries a full one-sided quadrature weight.
    original = np.arange(len(points))
    paired = np.concatenate((original, original[doubles]))
    back = np.concatenate((np.zeros(len(points), bool), np.ones(int(doubles.sum()), bool)))
    data = dict(positions=points[paired], normals=normals[paired] * np.where(back, -1, 1)[:, None],
                area_weights=np.full(len(paired), area / count),
                surface_ids=surface.face_surface[faces][paired], triangle_ids=faces[paired],
                categories=categories[paired], back_side=back)
    pilot_double = np.array([m['doubleSided'] for m in surface.materials], bool)[surface.face_material[pf]]
    stats = dict(raw_triangle_area=surface.raw_area, estimated_one_sided_visible_area=area,
                 estimated_two_sided_measure=surface.raw_area * float((pk * (1+pilot_double.astype(int))).mean()),
                 area_estimate_standard_error=surface.raw_area * math.sqrt(fraction*(1-fraction)/pilot_count),
                 area_pilot_count=pilot_count, area_pilot_alpha_rejected=int((~pk).sum()),
                 candidate_count=attempted, alpha_rejected_candidates=rejected,
                 base_sample_count=count, back_side_samples=int(back.sum()),
                 alpha_evaluation='Imported mip 0 alpha, point/bilinear; no view-dependent mip or LOD dithering')
    return data, stats
