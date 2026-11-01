// Editor-only A/B viewing. Only owned preview scene copies are modified.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering.Universal;
using UnityEngine.SceneManagement;

namespace LightOpt.Reproduction
{
    public sealed class LightOptOptimizationPreview : EditorWindow
    {
        [Serializable] public class LightEntry
        {
            public string id, path, type;
            public string changeStatus;
            public bool generated, dynamicState;
            public bool beforeEnabled = true, afterEnabled = true, referenceEnabled = true;
            public Vector3 position, direction;
            public Color colorRaw;
            public float range, innerSpotAngleDegrees, outerSpotAngleDegrees;
            public float referenceIntensity, beforeIntensity, afterIntensity;
            public Vector3 referencePosition, beforePosition, afterPosition;
            public float referenceRange, beforeRange, afterRange;
        }
        [Serializable] public class Experiment
        {
            public string name, family, sourceSceneGuid, sourceSceneSha256;
            public int steps;
            public float learningRate;
            public double initialLoss, finalLoss, initialTotalLoss, finalTotalLoss;
            public LightEntry[] lights;
        }
        [Serializable] public class Bundle
        {
            public int schemaVersion;
            public Experiment[] experiments;
            public LossRecord[] history;
            public LossWeights lossWeights;
            public string adjustmentSummary;
            public string[] adjustmentRounds;
        }
        [Serializable] public class LossRecord
        {
            public int step;
            public double reconstruction, overlap, attraction, total;
        }
        [Serializable] public class LossWeights { public double reconstruction, overlap, attraction; }

        [Serializable] public class RawLight
        {
            public string kind;
            public float[] position, direction, color_linear;
            public float intensity, range, inner_half_angle, outer_half_angle;
        }
        [Serializable] public class RawSummary { public LossRecord losses; }
        [Serializable] public class RawState { public RawLight[] lights; public RawSummary summary; }
        [Serializable] public class RawStates { public RawState reference, before, after; }
        [Serializable] public class Removal { public int[] removed_indices, kept_indices; }
        [Serializable] public class Addition { public int[] added_indices; }
        [Serializable] public class Stage
        {
            public string stage; public int round, light_count;
            public int[] removed_indices; public Removal removal; public Addition addition;
        }
        [Serializable] public class DynamicBundle
        {
            public string format, sourceSceneGuid, sourceSceneSha256, sampleFolder;
            public RawStates states; public Stage[] history; public LossWeights lossWeights;
        }
        [Serializable] public class SourceArray { public LightEntry[] items; }

        static Vector3 RawVector(float[] value)
        {
            Require(value != null && value.Length == 3, "Invalid saved vector.");
            var result = new Vector3(value[0], value[1], value[2]);
            Require(ValidPosition(result), "Non-finite saved vector.");
            return result;
        }

        static Bundle ConvertDynamic(string json)
        {
            var raw = JsonUtility.FromJson<DynamicBundle>(json);
            Require(raw.format == "lightopt.scene_light_set.v1" && raw.states?.reference?.lights != null &&
                raw.states.before?.lights != null && raw.states.after?.lights != null, "Invalid dynamic scene result.");
            var sources = JsonUtility.FromJson<SourceArray>("{\"items\":" + File.ReadAllText(Path.Combine(raw.sampleFolder, "lights.json")) + "}").items;
            Require(sources.Length == raw.states.reference.lights.Length && sources.Length == raw.states.before.lights.Length,
                "Reference/before source light count mismatch.");
            var entries = new List<LightEntry>();
            for (int i = 0; i < sources.Length; ++i)
            {
                var e = sources[i]; var r = raw.states.reference.lights[i]; var b = raw.states.before.lights[i];
                e.dynamicState = true; e.beforeEnabled = e.referenceEnabled = true; e.afterEnabled = false;
                e.referenceIntensity = r.intensity; e.beforeIntensity = b.intensity; e.afterIntensity = 0;
                e.referencePosition = e.afterPosition = RawVector(r.position); e.beforePosition = RawVector(b.position);
                e.referenceRange = e.afterRange = r.range; e.beforeRange = b.range;
                entries.Add(e);
            }
            // Final lights use separate preview-only objects; array indices are not persistent identities.
            for (int i = 0; i < raw.states.after.lights.Length; ++i)
            {
                var r = raw.states.after.lights[i]; var p = RawVector(r.position);
                var c = RawVector(r.color_linear);
                entries.Add(new LightEntry {
                    id = "final:" + i, path = "__LightOptFinal_" + i, generated = true, dynamicState = true,
                    beforeEnabled = false, referenceEnabled = false, afterEnabled = true,
                    type = r.kind, colorRaw = new Color(c.x, c.y, c.z, 1).gamma,
                    direction = r.kind == "Spot" ? RawVector(r.direction) : Vector3.forward,
                    position = p, referencePosition = p, beforePosition = p, afterPosition = p,
                    range = r.range, referenceRange = r.range, beforeRange = r.range, afterRange = r.range,
                    afterIntensity = r.intensity,
                    innerSpotAngleDegrees = 2 * r.inner_half_angle * Mathf.Rad2Deg,
                    outerSpotAngleDegrees = 2 * r.outer_half_angle * Mathf.Rad2Deg,
                });
            }
            int removed = 0, added = 0; var lines = new List<string>();
            foreach (var h in raw.history ?? Array.Empty<Stage>())
            {
                int rm = h.stage == "initial_removal" ? h.removed_indices?.Length ?? 0 : h.removal?.removed_indices?.Length ?? 0;
                int ad = h.addition?.added_indices?.Length ?? 0; removed += rm; added += ad;
                lines.Add((h.stage == "round" ? "Round " + (h.round + 1) : h.stage) +
                    ": removed " + rm + ", added " + ad + ", remaining " + h.light_count);
            }
            AnnotateLightChanges(entries, sources.Length, raw.history, raw.states.after.lights.Length);
            return new Bundle { schemaVersion = 3, lossWeights = raw.lossWeights,
                history = new[] { raw.states.before.summary.losses, raw.states.after.summary.losses },
                adjustmentSummary = "Before " + sources.Length + " / After " + raw.states.after.lights.Length +
                    " / Removed " + removed + " / Added " + added,
                adjustmentRounds = lines.ToArray(),
                experiments = new[] { new Experiment { name = Path.GetFileName(raw.sampleFolder), family = "scene",
                    sourceSceneGuid = raw.sourceSceneGuid, sourceSceneSha256 = raw.sourceSceneSha256,
                    lights = entries.ToArray() } } };
        }

        static void AnnotateLightChanges(List<LightEntry> entries, int sourceCount, Stage[] history, int finalCount)
        {
            // Replay list edits: indices belong to each stage's current list, not the source list.
            var origins = Enumerable.Range(0, sourceCount).ToList();
            var addedIn = Enumerable.Repeat("", sourceCount).ToList();
            foreach (var stage in history ?? Array.Empty<Stage>())
            {
                string when = stage.stage == "initial_removal" ? "initial pruning" : "Round " + (stage.round + 1);
                var removed = stage.stage == "initial_removal" ? stage.removed_indices : stage.removal?.removed_indices;
                var indices = removed ?? Array.Empty<int>();
                Require(indices.Distinct().Count() == indices.Length && indices.All(i => i >= 0 && i < origins.Count),
                    "Invalid removal indices in saved light history.");
                foreach (int index in indices)
                    if (origins[index] >= 0) entries[origins[index]].changeStatus = "REMOVED (" + when + ")";
                foreach (int index in indices.OrderByDescending(i => i))
                {
                    origins.RemoveAt(index);
                    addedIn.RemoveAt(index);
                }
                foreach (int index in stage.addition?.added_indices ?? Array.Empty<int>())
                {
                    Require(index == origins.Count, "Invalid added-light order in saved history.");
                    origins.Add(-1);
                    addedIn.Add(when);
                }
                Require(origins.Count == stage.light_count, "Saved light history count mismatch.");
            }
            Require(origins.Count == finalCount, "Cannot map final lights to the saved removal/addition history.");
            for (int i = 0; i < finalCount; ++i)
            {
                if (origins[i] >= 0)
                {
                    entries[origins[i]].changeStatus = "Retained -> final #" + (i + 1);
                    entries[sourceCount + i].changeStatus = "Optimized original #" + (origins[i] + 1);
                }
                else entries[sourceCount + i].changeStatus = "ADDED (" + addedIn[i] + ")";
            }
        }

        const string PreviewRoot = "Assets/LightOptPreviews/";
        [SerializeField] int selected;
        [SerializeField] SceneAsset sourceScene;
        [SerializeField] string dataFile, previewScenePath, familyKey, phase = "before";
        [SerializeField] bool lockCamera = true;
        [SerializeField] Camera fixedCamera;
        [SerializeField] Vector3 cameraPosition;
        [SerializeField] Quaternion cameraRotation;
        [SerializeField] float cameraFov, cameraAspect, cameraSize;
        [SerializeField] bool cameraOrthographic;
        Bundle bundle;
        Light[] boundLights;
        Vector2 scroll;
        string message;
        Experiment Current => bundle.experiments[selected];
        string CurrentFamily => Current.family + ":" + Current.sourceSceneGuid + ":" + Current.sourceSceneSha256;
        bool OwnSceneActive => !string.IsNullOrEmpty(previewScenePath) && SceneManager.GetActiveScene().path == previewScenePath && previewScenePath.StartsWith(PreviewRoot, StringComparison.Ordinal);
        bool Ready => bundle != null && OwnSceneActive && familyKey == CurrentFamily && boundLights != null && boundLights.Length == Current.lights.Length && boundLights.All(light => light != null);

        [MenuItem("Tools/LightOpt/Optimization preview")]
        public static void Open()
        {
            var window = GetWindow<LightOptOptimizationPreview>("LightOpt Comparison");
            window.minSize = new Vector2(740, 650);
            window.Show();
        }

        void OnEnable()
        {
            titleContent = new GUIContent("LightOpt Comparison");
            if (string.IsNullOrEmpty(dataFile) || dataFile.Replace('\\', '/').EndsWith("/LightOptPreviewData/experiments.json")) dataFile = Path.GetFullPath(Path.Combine(Application.dataPath, "../../../data/experiment.json"));
            if (File.Exists(dataFile)) TryUi(() => LoadData(dataFile));
        }

        void TryUi(Action action)
        {
            try { action(); message = null; }
            catch (Exception error) { message = error.Message; Debug.LogError("LightOpt preview: " + error.Message); }
        }

        public void LoadData(string file)
        {
            Require(!EditorApplication.isPlayingOrWillChangePlaymode, "Stop Play Mode before loading an experiment.");
            var json = File.ReadAllText(file);
            var loaded = JsonUtility.FromJson<Bundle>(json);
            if (loaded != null && loaded.schemaVersion == 3) loaded = ConvertDynamic(json);
            Require(loaded != null && (loaded.schemaVersion == 1 || loaded.schemaVersion == 2 || loaded.schemaVersion == 3) && loaded.experiments?.Length == 1, "Expected a single-scene experiment.json with schema version 1, 2, or 3.");
            foreach (var experiment in loaded.experiments)
            {
                Require(experiment.family == "scene", "Select an experiment.json generated by the scene optimizer.");
                Require(experiment.lights?.Length > 0, "The experiment contains no lights.");
                Require(experiment.lights.All(light => light != null && !string.IsNullOrEmpty(light.id) && !string.IsNullOrEmpty(light.path)), "A light identity or hierarchy path is missing.");
                Require(experiment.lights.Select(light => light.id).Distinct().Count() == experiment.lights.Length && experiment.lights.Select(light => light.path).Distinct().Count() == experiment.lights.Length, "The experiment contains duplicate light identities or paths.");
                foreach (var light in experiment.lights)
                {
                    if (loaded.schemaVersion != 3) light.beforeEnabled = light.afterEnabled = light.referenceEnabled = true;
                    Require(light.type == "Point" || light.type == "Spot", "Only Point and Spot lights are supported.");
                    Require(ValidIntensity(light.beforeIntensity) && ValidIntensity(light.afterIntensity) && ValidIntensity(light.referenceIntensity), "Light intensities must be finite and nonnegative.");
                    // Older experiments kept position and range fixed in every state.
                    if (loaded.schemaVersion == 1)
                    {
                        light.referencePosition = light.beforePosition = light.afterPosition = light.position;
                        light.referenceRange = light.beforeRange = light.afterRange = light.range;
                    }
                    Require(ValidPosition(light.referencePosition) && ValidPosition(light.beforePosition) && ValidPosition(light.afterPosition), "Light positions must be finite.");
                    Require(ValidRange(light.referenceRange) && ValidRange(light.beforeRange) && ValidRange(light.afterRange), "Light ranges must be finite and positive.");
                    Require((light.referencePosition - light.position).magnitude < 1e-4f && Near(light.referenceRange, light.range), "Reference position or range does not match the source parameters.");
                }
            }
            bundle = loaded;
            if (OwnSceneActive)
                foreach (var old in SceneManager.GetActiveScene().GetRootGameObjects()
                    .SelectMany(root => root.GetComponentsInChildren<Light>(true))
                    .Where(light => light.name.StartsWith("__LightOptFinal_", StringComparison.Ordinal))) old.enabled = false;
            dataFile = file;
            selected = 0;
            if (sourceScene == null) sourceScene = AssetDatabase.LoadAssetAtPath<SceneAsset>(AssetDatabase.GUIDToAssetPath(Current.sourceSceneGuid));
            boundLights = null;
            if (OwnSceneActive && familyKey == CurrentFamily) boundLights = Resolve(false);
        }

        void OnGUI()
        {
            EditorGUILayout.LabelField("Light Optimization Comparison", EditorStyles.boldLabel);
            EditorGUILayout.HelpBox("Compare intensity, position, and range in a preview copy using the same camera. Reference uses the original lights.", MessageType.Info);
            sourceScene = (SceneAsset)EditorGUILayout.ObjectField("Source Scene", sourceScene, typeof(SceneAsset), false);
            if (GUILayout.Button("Load Experiment"))
            {
                string file = EditorUtility.OpenFilePanel("Select Experiment JSON", Path.GetDirectoryName(dataFile), "json");
                if (!string.IsNullOrEmpty(file)) TryUi(() => LoadData(file));
            }
            if (bundle == null)
            {
                EditorGUILayout.HelpBox("No experiment loaded. Load data/experiment.json, or run the Python optimizer to create one.", MessageType.Warning);
                if (!string.IsNullOrEmpty(message)) EditorGUILayout.HelpBox(message, MessageType.Error);
                return;
            }
            EditorGUILayout.LabelField("Experiment File", dataFile);
            EditorGUILayout.LabelField("Sampled Scene", Current.name);
            if (GUILayout.Button("Generate / Open Preview", GUILayout.Height(30))) TryUi(() => Generate());
            using (new EditorGUI.DisabledScope(!Ready || EditorApplication.isPlayingOrWillChangePlaymode))
            {
                EditorGUILayout.BeginHorizontal();
                if (GUILayout.Button("Before", GUILayout.Height(30))) TryUi(() => Apply("before"));
                if (GUILayout.Button("After", GUILayout.Height(30))) TryUi(() => Apply("after"));
                if (GUILayout.Button("Reference", GUILayout.Height(30))) TryUi(() => Apply("reference"));
                EditorGUILayout.EndHorizontal();
                lockCamera = EditorGUILayout.Toggle("Lock Camera View", lockCamera);
                if (GUILayout.Button("Overview Camera (Entire Scene)")) TryUi(() => FrameOverview());
                if (GUILayout.Button("Use Current Scene View")) TryUi(UseSceneViewCamera);
                if (GUILayout.Button("Record Current Camera View")) TryUi(RecordCamera);
                if (GUILayout.Button("Enter Play Mode / Free Fly", GUILayout.Height(28))) TryUi(EnterNavigationPlay);
            }
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Current State", Ready ? PhaseName(phase) : "Generate or open the preview for this experiment.");
            DrawLossSummary();
            EditorGUILayout.LabelField("Light Count", Current.lights.Count(e => ExpectedEnabled(e, phase)).ToString());
            if (!string.IsNullOrEmpty(bundle.adjustmentSummary))
            {
                EditorGUILayout.LabelField("Light Set Changes", bundle.adjustmentSummary);
                foreach (var line in bundle.adjustmentRounds) EditorGUILayout.LabelField(line);
                EditorGUILayout.HelpBox("Rows include original lights and separate final preview objects. Status tracks removal/addition history; original lights disabled in After are not all removed.", MessageType.Info);
            }
            if (Ready)
            {
                EditorGUILayout.BeginHorizontal();
                GUILayout.Label("Light", GUILayout.Width(230));
                GUILayout.Label("Optimization Status", GUILayout.Width(240));
                foreach (string title in new[] { "Current", "Before", "After", "Reference" }) GUILayout.Label(title, GUILayout.Width(86));
                EditorGUILayout.EndHorizontal();
                scroll = EditorGUILayout.BeginScrollView(scroll);
                for (int i = 0; i < boundLights.Length; ++i)
                {
                    var entry = Current.lights[i];
                    EditorGUILayout.BeginHorizontal();
                    if (GUILayout.Button(new GUIContent((i + 1) + ". " + boundLights[i].name + " (" + entry.type + ")", entry.path), EditorStyles.linkLabel, GUILayout.Width(230))) Selection.activeGameObject = boundLights[i].gameObject;
                    GUILayout.Label(entry.changeStatus ?? "Fixed light set", GUILayout.Width(240));
                    foreach (float value in new[] { boundLights[i].intensity, entry.beforeIntensity, entry.afterIntensity, entry.referenceIntensity }) GUILayout.Label(value.ToString("G6"), GUILayout.Width(86));
                    EditorGUILayout.EndHorizontal();
                    EditorGUILayout.LabelField("Range: Before " + entry.beforeRange.ToString("G6") + " / After " + entry.afterRange.ToString("G6") + " / Reference " + entry.referenceRange.ToString("G6") + " / Current " + boundLights[i].range.ToString("G6"));
                    EditorGUILayout.LabelField("Position: Before " + entry.beforePosition.ToString("F3") + " / After " + entry.afterPosition.ToString("F3") + " / Reference " + entry.referencePosition.ToString("F3"));
                    EditorGUILayout.LabelField("Current Position", boundLights[i].transform.position.ToString("F3"));
                }
                EditorGUILayout.EndScrollView();
                if (boundLights.Where((light, i) => !Matches(light, Current.lights[i], phase)).Any())
                    EditorGUILayout.HelpBox("Light parameters changed outside this tool. Click a state button to restore them.", MessageType.Warning);
            }
            if (!string.IsNullOrEmpty(message)) EditorGUILayout.HelpBox(message, MessageType.Error);
        }

        LightOptRuntimeComparison.LossSnapshot GetLossSnapshot(bool after)
        {
            if (bundle.history == null || bundle.history.Length == 0) return null;
            var row = bundle.history[after ? bundle.history.Length - 1 : 0];
            if (row == null) return null;
            return new LightOptRuntimeComparison.LossSnapshot
            {
                reconstruction = row.reconstruction, overlap = row.overlap,
                attraction = row.attraction, total = row.total,
            };
        }

        void DrawLossSummary()
        {
            var before = GetLossSnapshot(false);
            var after = GetLossSnapshot(true);
            EditorGUILayout.LabelField("Saved Python losses (not live Unity image metrics)", EditorStyles.boldLabel);
            EditorGUILayout.BeginHorizontal();
            GUILayout.Label("Loss", GUILayout.Width(180));
            GUILayout.Label("Before", GUILayout.MinWidth(120));
            GUILayout.Label("After", GUILayout.MinWidth(120));
            EditorGUILayout.EndHorizontal();
            DrawLossRow("Reconstruction", before?.reconstruction ?? Current.initialLoss, after?.reconstruction ?? Current.finalLoss);
            DrawLossRow("Overlap", before?.overlap, after?.overlap);
            DrawLossRow("Attraction", before?.attraction, after?.attraction);
            DrawLossRow("Total", before?.total, after?.total);
            if (bundle.lossWeights != null)
                EditorGUILayout.LabelField("Weights", "Reconstruction: " + bundle.lossWeights.reconstruction.ToString("G4") + " / Overlap: " + bundle.lossWeights.overlap.ToString("G4") + " / Attraction: " + bundle.lossWeights.attraction.ToString("G4"));
        }

        static void DrawLossRow(string name, double? before, double? after)
        {
            EditorGUILayout.BeginHorizontal();
            GUILayout.Label(name, GUILayout.Width(180));
            GUILayout.Label(before.HasValue ? before.Value.ToString("G6") : "N/A", GUILayout.MinWidth(120));
            GUILayout.Label(after.HasValue ? after.Value.ToString("G6") : "N/A", GUILayout.MinWidth(120));
            EditorGUILayout.EndHorizontal();
        }

        public void SelectExperiment(int index)
        {
            Require(index >= 0 && index < bundle.experiments.Length, "Invalid experiment index.");
            selected = index;
            boundLights = null;
            if (OwnSceneActive && familyKey == CurrentFamily)
            {
                boundLights = Resolve(false);
                Apply(phase);
            }
        }

        public void Generate(bool showGameView = true)
        {
            Require(!EditorApplication.isPlayingOrWillChangePlaymode, "Stop Play Mode first.");
            Require(sourceScene != null && AssetDatabase.AssetPathToGUID(AssetDatabase.GetAssetPath(sourceScene)) == Current.sourceSceneGuid, "Select the source scene that matches this experiment.json.");
            Require(Hash(AssetDatabase.GetAssetPath(sourceScene)) == Current.sourceSceneSha256, "The source scene has changed. Export, sample, and optimize it again.");
            if (Ready)
            {
                Apply("before");
                if (showGameView) EditorApplication.ExecuteMenuItem("Window/General/Game");
                return;
            }
            for (int i = 0; i < SceneManager.sceneCount; ++i) Require(!SceneManager.GetSceneAt(i).isDirty, "Save unsaved scene changes before generating a preview.");
            string source = AssetDatabase.GetAssetPath(sourceScene);
            string folder = PreviewRoot + Current.family + "_" + Guid.NewGuid().ToString("N").Substring(0, 8);
            Directory.CreateDirectory(Path.Combine(Application.dataPath, "..", folder));
            AssetDatabase.Refresh();
            string path = folder + "/ScenePreview.unity";
            Require(AssetDatabase.CopyAsset(source, path), "Could not create the preview scene copy.");
            EditorSceneManager.OpenScene(path, OpenSceneMode.Single);
            previewScenePath = path;
            familyKey = CurrentFamily;
            boundLights = Resolve(true);
            FreezeComparedLightControllers();
            SetUpCamera();
            RecordCamera();
            Apply("before");
            if (showGameView) EditorApplication.ExecuteMenuItem("Window/General/Game");
            Repaint();
        }

        Light[] Resolve(bool requireReference)
        {
            var scene = SceneManager.GetActiveScene();
            Require(OwnSceneActive, "Open a preview scene generated by this tool first.");
            var all = scene.GetRootGameObjects().SelectMany(root => root.GetComponentsInChildren<Light>(true)).ToArray();
            string sceneGuid = AssetDatabase.AssetPathToGUID(scene.path);
            var result = new List<Light>();
            foreach (var entry in Current.lights)
            {
                Light light = entry.generated
                    ? all.SingleOrDefault(candidate => candidate.name == entry.path)
                    : all.SingleOrDefault(candidate => ObjectPath(candidate.transform) == entry.path);
                if (light == null && entry.generated)
                {
                    var go = new GameObject(entry.path);
                    SceneManager.MoveGameObjectToScene(go, scene);
                    light = go.AddComponent<Light>();
                    light.type = entry.type == "Spot" ? LightType.Spot : LightType.Point;
                    light.color = entry.colorRaw; light.range = entry.range;
                    light.transform.position = entry.position;
                    light.transform.forward = entry.direction.normalized;
                    if (light.type == LightType.Spot) { light.spotAngle = entry.outerSpotAngleDegrees; light.innerSpotAngle = entry.innerSpotAngleDegrees; }
                    light.intensity = 0; light.enabled = false;
                    Undo.RegisterCreatedObjectUndo(go, "LightOpt final preview light");
                }
                Require(light != null, "Could not resolve light: " + entry.path);
                if (entry.generated)
                {
                    Undo.RecordObjects(new UnityEngine.Object[] { light, light.transform }, "LightOpt final preview parameters");
                    light.type = entry.type == "Spot" ? LightType.Spot : LightType.Point;
                    light.color = entry.colorRaw;
                    light.transform.forward = entry.direction.normalized;
                    if (light.type == LightType.Spot) { light.spotAngle = entry.outerSpotAngleDegrees; light.innerSpotAngle = entry.innerSpotAngleDegrees; }
                }
                string id = GlobalObjectId.GetGlobalObjectIdSlow(light).ToString().Replace(sceneGuid, Current.sourceSceneGuid);
                Require(entry.generated || id == entry.id, "Light identity mismatch: " + entry.path);
                Require(light.type.ToString() == entry.type && (entry.dynamicState || light.enabled) && light.gameObject.activeInHierarchy, "The light type or active state has changed.");
                Require(Near(light.color.r, entry.colorRaw.r) && Near(light.color.g, entry.colorRaw.g) && Near(light.color.b, entry.colorRaw.b), "The light color has changed.");
                if (light.type == LightType.Spot) Require((light.transform.forward-entry.direction.normalized).magnitude < 1e-4f && Near(light.spotAngle, entry.outerSpotAngleDegrees) && Near(light.innerSpotAngle, entry.innerSpotAngleDegrees), "The Spot direction or cone angles have changed.");
                if (requireReference) Require(Matches(light, entry, "reference"), "Source light intensity, position, or range does not match the experiment.");
                result.Add(light);
            }
            return result.ToArray();
        }

        void FreezeComparedLightControllers()
        {
            LightOptCourtyardStaticLighting.Freeze(SceneManager.GetActiveScene());
            // These components can overwrite intensity or range during camera rendering.
            foreach (var light in boundLights)
                foreach (var behaviour in light.GetComponents<MonoBehaviour>())
                    if (behaviour != null && behaviour.GetType().Name == "DynamicLightController")
                    {
                        // The sample manager calls Reset even on disabled controllers during Scene camera renders.
                        Undo.DestroyObjectImmediate(behaviour);
                    }
                    else if (behaviour != null && new[] { "LightFlicker", "WashiLightController" }.Contains(behaviour.GetType().Name))
                    {
                        behaviour.enabled = false;
                        if (PrefabUtility.IsPartOfPrefabInstance(behaviour)) PrefabUtility.RecordPrefabInstancePropertyModifications(behaviour);
                    }
        }

        void SetUpCamera()
        {
            var cameras = SceneManager.GetActiveScene().GetRootGameObjects().SelectMany(root => root.GetComponentsInChildren<Camera>(true)).ToArray();
            fixedCamera = cameras.FirstOrDefault(camera => camera.enabled && camera.gameObject.activeInHierarchy && camera.CompareTag("MainCamera")) ?? cameras.FirstOrDefault(camera => camera.enabled && camera.gameObject.activeInHierarchy);
            bool createdCamera = fixedCamera == null;
            if (createdCamera)
            {
                var go = new GameObject("LightOpt Preview Camera");
                fixedCamera = go.AddComponent<Camera>();
                go.tag = "MainCamera";
                fixedCamera.clearFlags = CameraClearFlags.SolidColor;
                fixedCamera.backgroundColor = new Color(.025f, .025f, .025f);
                fixedCamera.GetUniversalAdditionalCameraData().renderPostProcessing = false;
            }
            foreach (var camera in cameras)
            {
                camera.enabled = camera == fixedCamera;
                if (PrefabUtility.IsPartOfPrefabInstance(camera)) PrefabUtility.RecordPrefabInstancePropertyModifications(camera);
            }
            if (createdCamera)
            {
                var positions = Current.lights.Select(light => light.position).ToArray();
                Vector3 center = positions.Aggregate(Vector3.zero, (a,b) => a+b)/positions.Length;
                float extent = Mathf.Max(2, positions.Max(position => (position-center).magnitude));
                fixedCamera.transform.position = center + new Vector3(1, 1, -1).normalized * extent * 3;
                fixedCamera.transform.LookAt(center);
                fixedCamera.fieldOfView = 55;
                fixedCamera.nearClipPlane = .1f;
                fixedCamera.farClipPlane = Mathf.Max(500, extent * 10);
            }
            foreach (var behaviour in fixedCamera.GetComponents<MonoBehaviour>())
                if (behaviour != null && behaviour.GetType().Name == "CinemachineBrain")
                {
                    behaviour.enabled = false;
                    if (PrefabUtility.IsPartOfPrefabInstance(behaviour)) PrefabUtility.RecordPrefabInstancePropertyModifications(behaviour);
                }
            if (PrefabUtility.IsPartOfPrefabInstance(fixedCamera)) PrefabUtility.RecordPrefabInstancePropertyModifications(fixedCamera);
            if (PrefabUtility.IsPartOfPrefabInstance(fixedCamera.transform)) PrefabUtility.RecordPrefabInstancePropertyModifications(fixedCamera.transform);
            InstallNavigationCamera();
            FrameOverview(false);
        }

        void InstallNavigationCamera()
        {
            Require(!EditorApplication.isPlayingOrWillChangePlaymode && OwnSceneActive && fixedCamera != null, "Open the preview scene in Edit Mode first.");
            if (fixedCamera.GetComponent<LightOptFlyCamera>() == null)
            {
                // Use a root camera so the original player rig cannot move its parent.
                var previous = fixedCamera;
                var go = new GameObject("LightOpt Navigation Camera");
                Undo.RegisterCreatedObjectUndo(go, "LightOpt navigation camera");
                SceneManager.MoveGameObjectToScene(go, SceneManager.GetActiveScene());
                var camera = go.AddComponent<Camera>();
                camera.CopyFrom(previous);
                camera.transform.SetPositionAndRotation(previous.transform.position, previous.transform.rotation);
                var previousData = previous.GetComponent<UniversalAdditionalCameraData>();
                var data = camera.GetUniversalAdditionalCameraData();
                if (previousData != null) EditorUtility.CopySerialized(previousData, data);
                data.renderType = CameraRenderType.Base;
                data.cameraStack?.Clear();
                camera.targetTexture = null;
                camera.enabled = true;
                go.AddComponent<LightOptFlyCamera>();
                go.AddComponent<AudioListener>();
                fixedCamera = camera;
            }
            foreach (var root in SceneManager.GetActiveScene().GetRootGameObjects())
            {
                foreach (var camera in root.GetComponentsInChildren<Camera>(true))
                {
                    Undo.RecordObject(camera, "LightOpt navigation camera");
                    camera.enabled = camera == fixedCamera;
                    if (camera.CompareTag("MainCamera") && camera != fixedCamera)
                    {
                        Undo.RecordObject(camera.gameObject, "LightOpt camera tag");
                        camera.gameObject.tag = "Untagged";
                        if (PrefabUtility.IsPartOfPrefabInstance(camera.gameObject)) PrefabUtility.RecordPrefabInstancePropertyModifications(camera.gameObject);
                    }
                    if (PrefabUtility.IsPartOfPrefabInstance(camera)) PrefabUtility.RecordPrefabInstancePropertyModifications(camera);
                }
                foreach (var listener in root.GetComponentsInChildren<AudioListener>(true))
                {
                    Undo.RecordObject(listener, "LightOpt navigation audio");
                    listener.enabled = listener.gameObject == fixedCamera.gameObject;
                    if (PrefabUtility.IsPartOfPrefabInstance(listener)) PrefabUtility.RecordPrefabInstancePropertyModifications(listener);
                }
                foreach (var behaviour in root.GetComponentsInChildren<MonoBehaviour>(true))
                    if (behaviour != null && new[] { "FirstPersonController", "StarterAssetsInputs", "PlayerInput", "CinemachineBrain", "SimpleCameraController" }.Contains(behaviour.GetType().Name))
                    {
                        Undo.RecordObject(behaviour, "LightOpt navigation input");
                        behaviour.enabled = false;
                        if (PrefabUtility.IsPartOfPrefabInstance(behaviour)) PrefabUtility.RecordPrefabInstancePropertyModifications(behaviour);
                    }
            }
            fixedCamera.gameObject.tag = "MainCamera";
            ConfigureRuntimeComparison();
            RecordCamera();
        }

        void ConfigureRuntimeComparison()
        {
            if (fixedCamera == null || fixedCamera.GetComponent<LightOptFlyCamera>() == null || boundLights == null) return;
            var comparison = fixedCamera.GetComponent<LightOptRuntimeComparison>();
            if (comparison == null) comparison = Undo.AddComponent<LightOptRuntimeComparison>(fixedCamera.gameObject);
            Undo.RecordObject(comparison, "LightOpt runtime comparison data");
            comparison.Configure(Current.lights.Select((entry, i) => new LightOptRuntimeComparison.Entry
            {
                light = boundLights[i],
                before = new LightOptRuntimeComparison.State { enabled = entry.beforeEnabled, intensity = entry.beforeIntensity, position = entry.beforePosition, range = entry.beforeRange },
                after = new LightOptRuntimeComparison.State { enabled = entry.afterEnabled, intensity = entry.afterIntensity, position = entry.afterPosition, range = entry.afterRange },
                reference = new LightOptRuntimeComparison.State { enabled = entry.referenceEnabled, intensity = entry.referenceIntensity, position = entry.referencePosition, range = entry.referenceRange },
            }).ToArray(), phase == "after" ? 1 : phase == "reference" ? 2 : 0,
                GetLossSnapshot(false), GetLossSnapshot(true), bundle.adjustmentSummary);
            EditorUtility.SetDirty(comparison);
        }

        void EnterNavigationPlay()
        {
            InstallNavigationCamera();
            FreezeComparedLightControllers();
            EditorSceneManager.MarkSceneDirty(SceneManager.GetActiveScene());
            Require(EditorSceneManager.SaveScene(SceneManager.GetActiveScene()), "Could not save the navigation preview.");
            EditorApplication.ExecuteMenuItem("Window/General/Game");
            EditorApplication.isPlaying = true;
        }

        public void FrameOverview(bool showGameView = true)
        {
            Require(!EditorApplication.isPlayingOrWillChangePlaymode && OwnSceneActive && fixedCamera != null, "Generate the preview in Edit Mode first.");
            var renderers = SceneManager.GetActiveScene().GetRootGameObjects()
                .SelectMany(root => root.GetComponentsInChildren<Renderer>(false))
                .Where(renderer => renderer.enabled && (renderer is MeshRenderer || renderer is SkinnedMeshRenderer))
                .Where(renderer => ValidPosition(renderer.bounds.center) && ValidPosition(renderer.bounds.extents) && renderer.bounds.extents.sqrMagnitude > 0)
                .ToArray();
            Require(renderers.Length > 0, "No active mesh renderers found. Use the current Scene View instead.");
            Bounds bounds = renderers[0].bounds;
            foreach (var renderer in renderers.Skip(1)) bounds.Encapsulate(renderer.bounds);

            Undo.RecordObjects(new UnityEngine.Object[] { fixedCamera, fixedCamera.transform }, "LightOpt overview camera");
            fixedCamera.orthographic = false;
            fixedCamera.usePhysicalProperties = false;
            fixedCamera.fieldOfView = 55;
            fixedCamera.ResetAspect();
            fixedCamera.nearClipPlane = .1f;
            Quaternion rotation = Quaternion.LookRotation(new Vector3(-1, -1.4f, 1).normalized, Vector3.up);
            Quaternion inverse = Quaternion.Inverse(rotation);
            float tanVertical = Mathf.Tan(.5f * fixedCamera.fieldOfView * Mathf.Deg2Rad);
            float tanHorizontal = tanVertical * Mathf.Max(.1f, fixedCamera.aspect);
            float distance = 1;
            // Fit all eight padded world-bounds corners within both field-of-view axes.
            for (int x = -1; x <= 1; x += 2)
                for (int y = -1; y <= 1; y += 2)
                    for (int z = -1; z <= 1; z += 2)
                    {
                        Vector3 corner = inverse * Vector3.Scale(bounds.extents * 1.1f, new Vector3(x, y, z));
                        distance = Mathf.Max(distance, Mathf.Abs(corner.x) / tanHorizontal - corner.z,
                            Mathf.Abs(corner.y) / tanVertical - corner.z, fixedCamera.nearClipPlane * 2 - corner.z);
                    }
            fixedCamera.transform.SetPositionAndRotation(bounds.center - rotation * Vector3.forward * distance, rotation);
            fixedCamera.farClipPlane = Mathf.Max(500, distance + bounds.extents.magnitude * 2 + 10);
            SaveCameraView(showGameView);
        }

        void UseSceneViewCamera()
        {
            Require(!EditorApplication.isPlayingOrWillChangePlaymode && OwnSceneActive && fixedCamera != null, "Generate the preview in Edit Mode first.");
            var view = SceneView.lastActiveSceneView;
            Require(view != null && view.camera != null, "Set the desired view in the Scene window first.");
            Undo.RecordObjects(new UnityEngine.Object[] { fixedCamera, fixedCamera.transform }, "LightOpt scene-view camera");
            fixedCamera.usePhysicalProperties = false;
            fixedCamera.transform.SetPositionAndRotation(view.camera.transform.position, view.camera.transform.rotation);
            fixedCamera.orthographic = view.camera.orthographic;
            fixedCamera.orthographicSize = view.camera.orthographicSize;
            fixedCamera.fieldOfView = view.camera.fieldOfView;
            fixedCamera.nearClipPlane = view.camera.nearClipPlane;
            fixedCamera.farClipPlane = view.camera.farClipPlane;
            fixedCamera.ResetAspect();
            SaveCameraView(true);
        }

        void SaveCameraView(bool showGameView)
        {
            if (PrefabUtility.IsPartOfPrefabInstance(fixedCamera)) PrefabUtility.RecordPrefabInstancePropertyModifications(fixedCamera);
            if (PrefabUtility.IsPartOfPrefabInstance(fixedCamera.transform)) PrefabUtility.RecordPrefabInstancePropertyModifications(fixedCamera.transform);
            RecordCamera();
            EditorSceneManager.MarkSceneDirty(SceneManager.GetActiveScene());
            Require(EditorSceneManager.SaveScene(SceneManager.GetActiveScene()), "Could not save the preview camera view.");
            if (showGameView) EditorApplication.ExecuteMenuItem("Window/General/Game");
            SceneView.RepaintAll();
            UnityEditorInternal.InternalEditorUtility.RepaintAllViews();
            Repaint();
        }

        public void RecordCamera()
        {
            Require(fixedCamera != null, "No preview camera found.");
            cameraPosition = fixedCamera.transform.position;
            cameraRotation = fixedCamera.transform.rotation;
            cameraFov = fixedCamera.fieldOfView;
            cameraAspect = fixedCamera.aspect;
            cameraOrthographic = fixedCamera.orthographic;
            cameraSize = fixedCamera.orthographicSize;
        }

        public void Apply(string targetPhase)
        {
            Require(!EditorApplication.isPlayingOrWillChangePlaymode && OwnSceneActive && familyKey == CurrentFamily, "Open the matching preview scene in Edit Mode.");
            Require(new[] { "before", "after", "reference" }.Contains(targetPhase), "Invalid comparison state.");
            boundLights = Resolve(false);
            Undo.IncrementCurrentGroup();
            int undoGroup = Undo.GetCurrentGroup();
            Undo.SetCurrentGroupName("LightOpt preview light parameters");
            Undo.RecordObjects(boundLights.SelectMany(light => new UnityEngine.Object[] { light, light.transform }).ToArray(), "LightOpt preview light parameters");
            for (int i = 0; i < boundLights.Length; ++i)
            {
                boundLights[i].intensity = Expected(Current.lights[i], targetPhase);
                boundLights[i].range = ExpectedRange(Current.lights[i], targetPhase);
                if (PrefabUtility.IsPartOfPrefabInstance(boundLights[i])) PrefabUtility.RecordPrefabInstancePropertyModifications(boundLights[i]);
            }
            // Set parents before children so every saved position remains a world-space position.
            foreach (int i in Enumerable.Range(0, boundLights.Length).OrderBy(i => ObjectPath(boundLights[i].transform).Count(c => c == '/')))
            {
                var transform = boundLights[i].transform;
                transform.position = ExpectedPosition(Current.lights[i], targetPhase);
                boundLights[i].enabled = ExpectedEnabled(Current.lights[i], targetPhase);
                if (PrefabUtility.IsPartOfPrefabInstance(transform)) PrefabUtility.RecordPrefabInstancePropertyModifications(transform);
            }
            Undo.FlushUndoRecordObjects();
            Undo.CollapseUndoOperations(undoGroup);
            if (lockCamera && fixedCamera != null)
            {
                fixedCamera.transform.SetPositionAndRotation(cameraPosition, cameraRotation);
                fixedCamera.fieldOfView = cameraFov;
                fixedCamera.aspect = cameraAspect;
                fixedCamera.orthographic = cameraOrthographic;
                fixedCamera.orthographicSize = cameraSize;
            }
            phase = targetPhase;
            ConfigureRuntimeComparison();
            EditorSceneManager.MarkSceneDirty(SceneManager.GetActiveScene());
            Require(EditorSceneManager.SaveScene(SceneManager.GetActiveScene()), "Could not save the preview scene.");
            SceneView.RepaintAll();
            UnityEditorInternal.InternalEditorUtility.RepaintAllViews();
            Repaint();
        }

        static float Expected(LightEntry light, string state) => state == "before" ? light.beforeIntensity : state == "after" ? light.afterIntensity : light.referenceIntensity;
        static Vector3 ExpectedPosition(LightEntry light, string state) => state == "before" ? light.beforePosition : state == "after" ? light.afterPosition : light.referencePosition;
        static float ExpectedRange(LightEntry light, string state) => state == "before" ? light.beforeRange : state == "after" ? light.afterRange : light.referenceRange;
        static bool ExpectedEnabled(LightEntry e, string state) => state == "before" ? e.beforeEnabled : state == "after" ? e.afterEnabled : e.referenceEnabled;
        static bool Matches(Light light, LightEntry entry, string state) => light.enabled == ExpectedEnabled(entry, state) && Near(light.intensity, Expected(entry, state)) && Near(light.range, ExpectedRange(entry, state)) && (light.transform.position - ExpectedPosition(entry, state)).magnitude < 1e-4f;
        static bool ValidIntensity(float value) => !float.IsNaN(value) && !float.IsInfinity(value) && value >= 0;
        static bool ValidRange(float value) => ValidIntensity(value) && value > 0;
        static bool ValidPosition(Vector3 value) => !float.IsNaN(value.x) && !float.IsInfinity(value.x) && !float.IsNaN(value.y) && !float.IsInfinity(value.y) && !float.IsNaN(value.z) && !float.IsInfinity(value.z);
        static bool Near(float a, float b) => Mathf.Abs(a-b) <= 1e-5f*Mathf.Max(1, Mathf.Abs(b));
        static string PhaseName(string value) => value == "before" ? "Before" : value == "after" ? "After" : "Reference";
        static void Require(bool ok, string error) { if (!ok) throw new InvalidOperationException(error); }
        static string ObjectPath(Transform node)
        {
            var parts = new List<string>();
            while (node != null) { parts.Add(node.name + "[" + node.GetSiblingIndex() + "]"); node = node.parent; }
            parts.Reverse(); return string.Join("/", parts);
        }
        static string Hash(string file)
        {
            using (var sha = SHA256.Create())
            using (var stream = File.OpenRead(file)) return BitConverter.ToString(sha.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
        }

        // Batch validation exercises the same generation and switching methods as the UI.
        [Serializable] public class CheckResult { public string experiment, scene; public int lights; public bool allStatesMatch, cameraUnchanged; }
        [Serializable] public class CheckReport { public bool passed, originalSceneUnchanged; public List<CheckResult> cases = new List<CheckResult>(); }
        public static void VerifyFromCommandLine()
        {
            string[] args = Environment.GetCommandLineArgs();
            int index = Array.IndexOf(args, "-lightoptPreviewReport");
            Require(index >= 0 && index+1 < args.Length, "Missing -lightoptPreviewReport");
            var setup = EditorSceneManager.GetSceneManagerSetup();
            var window = CreateInstance<LightOptOptimizationPreview>();
            int dataIndex = Array.IndexOf(args, "-lightoptExperiment");
            if (dataIndex >= 0 && dataIndex + 1 < args.Length) window.LoadData(args[dataIndex + 1]);
            Require(window.bundle != null, "Run optimize_scene.py before validation");
            var report = new CheckReport();
            string source = AssetDatabase.GUIDToAssetPath(window.Current.sourceSceneGuid);
            string originalHash = Hash(source);
            string gardenPath = null;
            try
            {
                for (int i = 0; i < window.bundle.experiments.Length; ++i)
                {
                    window.SelectExperiment(i);
                    window.Generate(false);
                    Vector3 position = window.fixedCamera.transform.position;
                    Quaternion rotation = window.fixedCamera.transform.rotation;
                    foreach (string state in new[] { "before", "after", "reference", "before", "after" })
                    {
                        window.Apply(state);
                        Require(window.boundLights.Where((light,n) => !Matches(light, window.Current.lights[n], state)).Count() == 0, "Intensity, position, or range switch failed");
                        Require(window.fixedCamera.transform.position == position && window.fixedCamera.transform.rotation == rotation, "Camera changed while switching");
                    }
                    if (window.Current.family != "synthetic")
                    {
                        if (gardenPath != null) Require(gardenPath == window.previewScenePath, "Garden experiments should reuse the same preview scene");
                        gardenPath = window.previewScenePath;
                    }
                    report.cases.Add(new CheckResult { experiment = window.Current.name, scene = window.previewScenePath,
                        lights = window.boundLights.Length, allStatesMatch = true, cameraUnchanged = true });
                }
                report.originalSceneUnchanged = Hash(source) == originalHash;
                Require(report.originalSceneUnchanged, "Original Garden scene changed");
                report.passed = true;
                File.WriteAllText(args[index+1], JsonUtility.ToJson(report, true));
                Debug.Log("LightOpt preview checks passed: selected scene, intensity/position/range states and fixed camera.");
            }
            finally
            {
                DestroyImmediate(window);
                if (setup.Length > 0) EditorSceneManager.RestoreSceneManagerSetup(setup);
            }
        }
    }
}
