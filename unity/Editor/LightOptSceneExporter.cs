// Install in Assets/Editor. Export only; never changes scene objects or import settings.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Unity.Collections;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.SceneManagement;

namespace LightOpt.Reproduction
{
    public static class LightOptSceneExporter
    {
        [Serializable] public class MeshRecord
        {
            public string path;
            public int layer, vertexStart, vertexCount, triangleStart, triangleCount;
        }
        [Serializable] public class SurfaceRecord
        {
            public string path;
            public int triangleStart, triangleCount, materialIndex;
        }
        [Serializable] public class MaterialRecord
        {
            public string name, shader, alphaMode, alphaMaskFile, wrapU, wrapV, filter;
            public int width, height, category;
            public float cutoff, alphaMultiplier = 1;
            public bool doubleSided;
            public Vector2 uvScale = Vector2.one, uvOffset = Vector2.zero;
        }
        [Serializable] public class LightRecord
        {
            public string id, path, type, mode, shadows;
            public bool enabled, active, hasCookie, useColorTemperature;
            public int cullingMask;
            public Vector3 position, direction;
            public Color colorRaw, colorLinear;
            public float intensity, range, innerSpotAngleDegrees, outerSpotAngleDegrees, colorTemperature;
        }
        [Serializable] public class SceneRecord
        {
            public int schemaVersion = 2;
            public string scene, unityVersion, colorSpace;
            public string coordinates = "Unity world: x right, y up, z forward; no axis conversion";
            public string geometryScope = "Active-scene enabled MeshRenderers, highest LOD, opaque and supported alpha-clipped submeshes";
            public string vertexLayout = "vertices.f32: little-endian float32 [x,y,z,nx,ny,nz]";
            public string triangleLayout = "triangles.i32: little-endian int32 global vertex indices, three per triangle";
            public string uvLayout = "uvs.f32: little-endian float32 UV0 [u,v] per vertex";
            public List<MaterialRecord> materials = new List<MaterialRecord>();
            public List<SurfaceRecord> surfaces = new List<SurfaceRecord>();
            public List<MeshRecord> meshes = new List<MeshRecord>();
            public List<LightRecord> lights = new List<LightRecord>();
            public List<string> notes = new List<string>();
        }

        [MenuItem("Tools/LightOpt/Export active scene")]
        public static void ExportActiveScene()
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode)
                throw new InvalidOperationException("Stop Play Mode before exporting.");
            var scene = SceneManager.GetActiveScene();
            if (!scene.IsValid() || !scene.isLoaded)
                throw new InvalidOperationException("No active scene.");
            var parent = Path.GetDirectoryName(Application.dataPath);
            var folder = EditorUtility.SaveFolderPanel("New LightOpt export folder", parent, "");
            if (string.IsNullOrEmpty(folder)) return;
            // Avoid mixing stale and new datasets; select a fresh, empty folder.
            if (Directory.GetFileSystemEntries(folder).Length != 0)
                throw new InvalidOperationException("Choose a new empty folder for this export.");
            var result = new SceneRecord { scene = scene.path, unityVersion = Application.unityVersion,
                colorSpace = QualitySettings.activeColorSpace.ToString() };
            var roots = scene.GetRootGameObjects();
            var nonFirstLod = new HashSet<Renderer>();
            var firstLod = new HashSet<Renderer>();
            foreach (var group in roots.SelectMany(r => r.GetComponentsInChildren<LODGroup>(true)))
            {
                var lods = group.GetLODs();
                for (int i = 0; i < lods.Length; ++i)
                    foreach (var renderer in lods[i].renderers)
                        if (renderer != null)
                        { if (i == 0) firstLod.Add(renderer); else nonFirstLod.Add(renderer); }
            }
            nonFirstLod.ExceptWith(firstLod);
            int vertexCount = 0, triangleCount = 0;
            var materialIds = new Dictionary<Material, int>();
            try
            {
                using (var verticesFile = new BinaryWriter(File.Create(Path.Combine(folder, "vertices.f32"))))
                using (var uvsFile = new BinaryWriter(File.Create(Path.Combine(folder, "uvs.f32"))))
                using (var trianglesFile = new BinaryWriter(File.Create(Path.Combine(folder, "triangles.i32"))))
                {
                    var renderers = roots.SelectMany(r => r.GetComponentsInChildren<MeshRenderer>(true))
                        .OrderBy(r => ObjectPath(r.transform), StringComparer.Ordinal).ToArray();
                    for (int i = 0; i < renderers.Length; ++i)
                    {
                        var renderer = renderers[i];
                        if (!renderer.enabled || !renderer.gameObject.activeInHierarchy || nonFirstLod.Contains(renderer)) continue;
                        if (EditorUtility.DisplayCancelableProgressBar("LightOpt export", renderer.name,
                            (float)i / Math.Max(1, renderers.Length)))
                            throw new OperationCanceledException("Export cancelled; delete incomplete export and retry.");
                        var filter = renderer.GetComponent<MeshFilter>();
                        if (filter == null || filter.sharedMesh == null) continue;
                        var mesh = filter.sharedMesh;
                        var materials = renderer.sharedMaterials;
                        using (var dataArray = MeshUtility.AcquireReadOnlyMeshData(mesh))
                        {
                            var data = dataArray[0];
                            var accepted = new List<int>();
                            for (int sub = 0; sub < data.subMeshCount; ++sub)
                            {
                                var material = sub < materials.Length ? materials[sub] : null;
                                if (material == null || data.GetSubMesh(sub).topology != MeshTopology.Triangles)
                                { result.notes.Add("Excluded invalid submesh: " + ObjectPath(renderer.transform) + " #" + sub); continue; }
                                if (renderer.HasPropertyBlock())
                                    throw new InvalidOperationException("MaterialPropertyBlock needs explicit support: " + ObjectPath(renderer.transform));
                                if (!materialIds.TryGetValue(material, out int materialId))
                                {
                                    var record = DescribeMaterial(material, folder, result);
                                    materialId = record == null ? -1 : result.materials.Count;
                                    if (record != null) result.materials.Add(record);
                                    materialIds.Add(material, materialId);
                                }
                                if (materialId < 0)
                                { result.notes.Add("Excluded unsupported material: " + ObjectPath(renderer.transform) + " #" + sub + " " + material.name); continue; }
                                if (result.materials[materialId].alphaMode == "clip" && !data.HasVertexAttribute(VertexAttribute.TexCoord0))
                                    throw new InvalidOperationException("Alpha-clipped mesh has no UV0: " + renderer.name);
                                result.surfaces.Add(new SurfaceRecord { path = ObjectPath(renderer.transform) + " #" + sub,
                                    triangleStart = triangleCount + accepted.Count / 3,
                                    triangleCount = (int)data.GetSubMesh(sub).indexCount / 3, materialIndex = materialId });
                                using (var indices = new NativeArray<int>((int)data.GetSubMesh(sub).indexCount, Allocator.Temp))
                                {
                                    data.GetIndices(indices, sub, true);
                                    foreach (int index in indices) accepted.Add(index);
                                }
                            }
                            if (accepted.Count == 0) continue;
                            using (var positions = new NativeArray<Vector3>(data.vertexCount, Allocator.Temp))
                            using (var normals = new NativeArray<Vector3>(data.vertexCount, Allocator.Temp))
                            using (var uvs = new NativeArray<Vector2>(data.vertexCount, Allocator.Temp))
                            {
                                data.GetVertices(positions);
                                if (data.HasVertexAttribute(VertexAttribute.TexCoord0)) data.GetUVs(0, uvs);
                                bool hasNormals = data.HasVertexAttribute(VertexAttribute.Normal);
                                if (hasNormals) data.GetNormals(normals);
                                else throw new InvalidOperationException("Mesh has no normals: " + ObjectPath(renderer.transform));
                                Matrix4x4 matrix = renderer.localToWorldMatrix;
                                if (Math.Abs(matrix.determinant) < 1e-12f)
                                    throw new InvalidOperationException("Singular object transform: " + renderer.name);
                                Matrix4x4 normalMatrix = matrix.inverse.transpose;
                                for (int v = 0; v < data.vertexCount; ++v)
                                {
                                    Write(verticesFile, matrix.MultiplyPoint3x4(positions[v]));
                                    Vector3 normal = normalMatrix.MultiplyVector(normals[v]).normalized;
                                    if (normal.sqrMagnitude < 0.5f)
                                        throw new InvalidOperationException("Invalid mesh normal: " + renderer.name);
                                    Write(verticesFile, normal);
                                    uvsFile.Write(uvs[v].x); uvsFile.Write(uvs[v].y);
                                }
                                foreach (int index in accepted) trianglesFile.Write(checked(index + vertexCount));
                                result.meshes.Add(new MeshRecord { path = ObjectPath(renderer.transform),
                                    layer = renderer.gameObject.layer, vertexStart = vertexCount, vertexCount = data.vertexCount,
                                    triangleStart = triangleCount, triangleCount = accepted.Count / 3 });
                                vertexCount = checked(vertexCount + data.vertexCount);
                                triangleCount += accepted.Count / 3;
                            }
                        }
                    }
                }
                foreach (var light in roots.SelectMany(r => r.GetComponentsInChildren<Light>(true)))
                    result.lights.Add(new LightRecord {
                        id = GlobalObjectId.GetGlobalObjectIdSlow(light).ToString(), path = ObjectPath(light.transform),
                        type = light.type.ToString(), mode = light.lightmapBakeType.ToString(),
                        enabled = light.enabled, active = light.gameObject.activeInHierarchy,
                        position = light.transform.position, direction = light.transform.forward,
                        colorRaw = light.color, colorLinear = light.color.linear,
                        intensity = light.intensity, range = light.range,
                        innerSpotAngleDegrees = light.innerSpotAngle, outerSpotAngleDegrees = light.spotAngle,
                        shadows = light.shadows.ToString(), hasCookie = light.cookie != null,
                        cullingMask = light.cullingMask, useColorTemperature = light.useColorTemperature,
                        colorTemperature = light.colorTemperature });
                foreach (var root in roots)
                {
                    foreach (var terrain in root.GetComponentsInChildren<Terrain>(true))
                        result.notes.Add("Terrain not exported: " + ObjectPath(terrain.transform));
                    foreach (var skin in root.GetComponentsInChildren<SkinnedMeshRenderer>(true))
                        result.notes.Add("Skinned mesh not exported: " + ObjectPath(skin.transform));
                }
                result.notes.Add("No visibility/shadow evaluation. No terrain, particles, decals, or shader vertex displacement.");
                result.notes.Add("Normals are inverse-transpose transformed vertex normals; tangents/normal maps are not exported.");
                result.notes.Add("No visibility filtering of buried/back surfaces. No scene or light settings modified. Freeze wind/vertex animation in Unity for comparisons.");
                File.WriteAllText(Path.Combine(folder, "scene.json"), JsonUtility.ToJson(result, true));
                Debug.Log($"LightOpt export complete: {result.meshes.Count} meshes, {triangleCount} triangles, " +
                    $"{result.lights.Count} lights. Output: {folder}");
                EditorUtility.RevealInFinder(Path.Combine(folder, "scene.json"));
            }
            catch
            {
                File.WriteAllText(Path.Combine(folder, "INCOMPLETE.txt"), "Export failed or cancelled. Do not sample this folder.");
                throw;
            }
            finally { EditorUtility.ClearProgressBar(); }
        }
        // Garden graph was inspected: BaseMap.A -> Alpha, constant threshold .5,
        // RenderFace Both, UV0 without tiling/offset; material override is disabled.
        static MaterialRecord DescribeMaterial(Material material, string folder, SceneRecord scene)
        {
            string shaderPath = AssetDatabase.GetAssetPath(material.shader).Replace('\\', '/');
            bool garden = AssetDatabase.AssetPathToGUID(shaderPath) == "1c4cbaedbae058f47af56a8d4937a4d5";
            string shaderName = material.shader.name;
            bool standard = shaderName == "Universal Render Pipeline/Lit" ||
                shaderName == "Universal Render Pipeline/Simple Lit" || shaderName == "Standard" ||
                shaderName == "Legacy Shaders/Transparent/Cutout/Diffuse";
            if ((material.HasProperty("_Surface") && material.GetFloat("_Surface") > 0) ||
                material.renderQueue >= 3000 || material.GetTag("RenderType", false) == "Transparent")
                return null;
            bool clip = garden || material.IsKeywordEnabled("_ALPHATEST_ON") ||
                (material.HasProperty("_AlphaClip") && material.GetFloat("_AlphaClip") > .5f) ||
                material.GetTag("RenderType", false) == "TransparentCutout";
            if (!garden && !standard && (clip || material.renderQueue >= 2450))
            {
                scene.notes.Add("Unsupported cutout shader: " + material.name + " / " + shaderName);
                return null;
            }
            var record = new MaterialRecord { name = material.name, shader = shaderName,
                alphaMode = clip ? "clip" : "opaque", cutoff = garden ? .5f :
                    (material.HasProperty("_Cutoff") ? material.GetFloat("_Cutoff") : .5f),
                doubleSided = garden || (material.HasProperty("_Cull") && material.GetFloat("_Cull") == 0),
                category = garden ? (material.name.IndexOf("Bark", StringComparison.OrdinalIgnoreCase) >= 0 ||
                    material.name.IndexOf("Knot", StringComparison.OrdinalIgnoreCase) >= 0 ? 1 : 2) : 0 };
            if (!clip) return record;
            string prop = material.HasProperty("_BaseMap") ? "_BaseMap" : "_MainTex";
            var texture = material.HasProperty(prop) ? material.GetTexture(prop) as Texture2D : null;
            if (!garden)
            {
                record.uvScale = material.GetTextureScale(prop); record.uvOffset = material.GetTextureOffset(prop);
                string color = material.HasProperty("_BaseColor") ? "_BaseColor" : "_Color";
                if (material.HasProperty(color)) record.alphaMultiplier = material.GetColor(color).a;
            }
            if (texture != null)
            {
                record.width = texture.width; record.height = texture.height;
                record.wrapU = texture.wrapModeU.ToString(); record.wrapV = texture.wrapModeV.ToString();
                record.filter = texture.filterMode == FilterMode.Point ? "point" : "bilinear";
                string guid = AssetDatabase.AssetPathToGUID(AssetDatabase.GetAssetPath(texture));
                record.alphaMaskFile = "alpha/" + (string.IsNullOrEmpty(guid) ? Guid.NewGuid().ToString("N") : guid) + ".a8";
                string target = Path.Combine(folder, record.alphaMaskFile);
                if (!File.Exists(target)) ExportAlpha(texture, target);
            }
            if (garden) scene.notes.Add("Garden vegetation adapter: " + material.name +
                "; BaseMap alpha >= .5, both faces, static undeformed mesh. Subsurface transmission is not modeled.");
            return record;
        }
        static void ExportAlpha(Texture2D source, string target)
        {
            // GPU readback preserves imported alpha without changing Read/Write or import settings.
            var previous = RenderTexture.active;
            bool previousSrgb = GL.sRGBWrite;
            var rt = RenderTexture.GetTemporary(source.width, source.height, 0, RenderTextureFormat.ARGB32,
                RenderTextureReadWrite.Linear);
            Texture2D copy = null;
            try
            {
                GL.sRGBWrite = false;
                Graphics.Blit(source, rt);
                RenderTexture.active = rt;
                copy = new Texture2D(source.width, source.height, TextureFormat.RGBA32, false, true);
                copy.ReadPixels(new Rect(0, 0, source.width, source.height), 0, 0, false);
                copy.Apply(false, false);
                var pixels = copy.GetPixels32();
                var alpha = new byte[pixels.Length];
                for (int i = 0; i < pixels.Length; ++i) alpha[i] = pixels[i].a;
                Directory.CreateDirectory(Path.GetDirectoryName(target));
                File.WriteAllBytes(target, alpha); // bottom row first, x fastest
            }
            finally
            {
                RenderTexture.active = previous; GL.sRGBWrite = previousSrgb;
                if (copy != null) UnityEngine.Object.DestroyImmediate(copy);
                RenderTexture.ReleaseTemporary(rt);
            }
        }
        static string ObjectPath(Transform node)
        {
            var names = new List<string>();
            while (node != null) { names.Add(node.name + "[" + node.GetSiblingIndex() + "]"); node = node.parent; }
            names.Reverse(); return string.Join("/", names);
        }
        static void Write(BinaryWriter writer, Vector3 v)
        { writer.Write(v.x); writer.Write(v.y); writer.Write(v.z); }
    }
}
