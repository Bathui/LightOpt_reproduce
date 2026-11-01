using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

public static class LightOptCourtyardStaticLighting
{
    [Serializable] public class Entry { public string path, type; public bool wasEnabled; }
    [Serializable] public class Report
    {
        public string scene;
        public int enabledBefore, enabledAfter, hiddenControls, frozenBodies;
        public bool lightingPreserved;
        public List<Entry> controllers = new List<Entry>();
    }

    [MenuItem("Tools/LightOpt/Disable Courtyard Day Night Completely")]
    public static void FreezeActiveScene()
    {
        if (EditorApplication.isPlayingOrWillChangePlaymode)
            throw new InvalidOperationException("Stop Play Mode before preparing static lighting.");
        var scene = SceneManager.GetActiveScene();
        var report = Freeze(scene);
        if (report == null) throw new InvalidOperationException("Open Courtyard_LightOpt or its comparison scene.");
        EditorSceneManager.MarkSceneDirty(scene);
        EditorSceneManager.SaveScene(scene);
        string project = Directory.GetParent(Application.dataPath).FullName;
        string output = Path.GetFullPath(Path.Combine(project, "../../tmp/review/courtyard-disable-day-night"));
        Directory.CreateDirectory(output);
        File.WriteAllText(Path.Combine(output, "freeze-report.json"), JsonUtility.ToJson(report, true));
    }

    public static Report Freeze(Scene scene)
    {
        if (!scene.IsValid() || !scene.isLoaded) return null;
        var roots = scene.GetRootGameObjects();
        var behaviours = roots.SelectMany(root => root.GetComponentsInChildren<MonoBehaviour>(true)).Where(x => x != null).ToArray();
        var managers = behaviours.OfType<TimeOfDayManager>().ToArray();
        if (managers.Length == 0 || (scene.name != "Courtyard_LightOpt" &&
            !scene.path.StartsWith("Assets/LightOptPreviews/", StringComparison.Ordinal))) return null;
        var controllers = new List<Behaviour>();
        var controls = new HashSet<GameObject>();
        var report = new Report { scene = scene.path };
        foreach (var item in behaviours)
            if (item is ObjectProperties || new[] { "TimeOfDayManager", "AnimationHub", "EnvironmentUpdater",
                "LightTimer", "LightTimerOverride", "LightAlwaysOn", "Guardian", "GuardianStatic",
                "AgentTween", "LightFlicker", "ColorSwatch", "CopySceneView", "CameraStealer", "FlashLight",
                "FirstPersonController", "RigidbodyFirstPersonController" }.Contains(item.GetType().Name))
                controllers.Add(item);
        foreach (GameObject root in roots)
        {
            controllers.AddRange(root.GetComponentsInChildren<Animation>(true));
            controllers.AddRange(root.GetComponentsInChildren<Animator>(true));
            controllers.AddRange(root.GetComponentsInChildren<UnityEngine.AI.NavMeshAgent>(true));
        }
        foreach (var manager in managers)
        {
            if (manager.timeSlider != null) controls.Add(manager.timeSlider.gameObject);
            if (manager.speedSlider != null) controls.Add(manager.speedSlider.gameObject);
            if (manager.autoCycleTimeToggle != null) controls.Add(manager.autoCycleTimeToggle.gameObject);
        }
        var objects = roots.SelectMany(root => root.GetComponentsInChildren<Transform>(true)).ToArray();
        var lights = roots.SelectMany(root => root.GetComponentsInChildren<Light>(true)).ToArray();
        var renderers = roots.SelectMany(root => root.GetComponentsInChildren<Renderer>(true)).ToArray();
        // Legacy OnDisable handlers can reset guardians, lights or the camera. Preserve the current scene state.
        var restore = new List<Action>();
        foreach (var transform in objects)
        {
            var item = transform; var p = item.localPosition; var r = item.localRotation; var s = item.localScale;
            bool active = item.gameObject.activeSelf;
            restore.Add(() => {
                if (item.localPosition != p) item.localPosition = p;
                if (item.localRotation != r) item.localRotation = r;
                if (item.localScale != s) item.localScale = s;
                if (item.gameObject.activeSelf != active) item.gameObject.SetActive(active);
            });
        }
        foreach (var light in lights)
        {
            var item = light; bool enabled = item.enabled; Color color = item.color;
            float intensity = item.intensity, range = item.range, inner = item.innerSpotAngle, outer = item.spotAngle;
            restore.Add(() => { item.enabled = enabled; item.color = color; item.intensity = intensity;
                item.range = range; item.innerSpotAngle = inner; item.spotAngle = outer; });
        }
        foreach (var renderer in renderers)
        { var item = renderer; bool enabled = item.enabled; restore.Add(() => item.enabled = enabled); }
        foreach (var camera in roots.SelectMany(root => root.GetComponentsInChildren<Camera>(true)))
        { var item = camera; bool enabled = item.enabled; restore.Add(() => item.enabled = enabled); }
        var mode = RenderSettings.ambientMode; var ambient = RenderSettings.ambientLight;
        var skyColor = RenderSettings.ambientSkyColor; var equator = RenderSettings.ambientEquatorColor;
        var ground = RenderSettings.ambientGroundColor; float ambientIntensity = RenderSettings.ambientIntensity;
        var sky = RenderSettings.skybox; float exposure = sky != null && sky.HasProperty("_Exposure") ? sky.GetFloat("_Exposure") : 0;
        var before = LightSignature(lights);
        foreach (var controller in controllers.Distinct())
        {
            report.controllers.Add(new Entry { path = PathOf(controller.transform), type = controller.GetType().Name, wasEnabled = controller.enabled });
            if (controller.enabled) report.enabledBefore++;
            Undo.RecordObject(controller, "Disable Courtyard day/night driver");
            controller.enabled = false;
            var manager = controller as TimeOfDayManager;
            if (manager != null) { manager.autoCycleTime = false; manager.animationControl = false; }
            var hub = controller as AnimationHub;
            if (hub != null) hub.runInEditMode = false;
            if (PrefabUtility.IsPartOfPrefabInstance(controller)) PrefabUtility.RecordPrefabInstancePropertyModifications(controller);
        }
        // One driver's OnDisable can re-enable another driver; settle those effects before restoring lights.
        for (int pass = 0; pass < 4; ++pass)
            foreach (var controller in controllers)
                if (controller != null && controller.enabled) controller.enabled = false;
        foreach (Action action in restore) action();
        RenderSettings.ambientMode = mode; RenderSettings.ambientLight = ambient;
        RenderSettings.ambientSkyColor = skyColor; RenderSettings.ambientEquatorColor = equator;
        RenderSettings.ambientGroundColor = ground; RenderSettings.ambientIntensity = ambientIntensity;
        RenderSettings.skybox = sky;
        if (sky != null && sky.HasProperty("_Exposure")) sky.SetFloat("_Exposure", exposure);
        var marker = behaviours.OfType<LightOptStaticLighting>().FirstOrDefault();
        if (marker == null)
        {
            var root = new GameObject("__LightOpt_StaticLighting");
            SceneManager.MoveGameObjectToScene(root, scene);
            marker = root.AddComponent<LightOptStaticLighting>();
        }
        marker.controllers = controllers.Distinct().ToArray();
        marker.timeControls = controls.ToArray();
        marker.bodies = roots.SelectMany(root => root.GetComponentsInChildren<Rigidbody>(true)).ToArray();
        report.frozenBodies = marker.bodies.Length;
        marker.Enforce();
        EditorUtility.SetDirty(marker);
        report.enabledAfter = marker.controllers.Count(controller => controller != null && controller.enabled);
        report.hiddenControls = controls.Count;
        report.lightingPreserved = before == LightSignature(lights);
        if (!report.lightingPreserved)
        {
            string directory = Path.GetFullPath(Path.Combine(Application.dataPath, "../../../tmp/review/courtyard-disable-day-night"));
            File.WriteAllText(Path.Combine(directory, "lights-before.txt"), before);
            File.WriteAllText(Path.Combine(directory, "lights-after.txt"), LightSignature(lights));
            throw new InvalidOperationException("Light state changed while disabling legacy controllers.");
        }
        SceneView.RepaintAll();
        return report;
    }

    static string PathOf(Transform item) { return item.parent == null ? item.name : PathOf(item.parent) + "/" + item.name; }

    static string LightSignature(IEnumerable<Light> lights)
    {
        return string.Join("\n", lights.Select(light => JsonUtility.ToJson(new LightState {
            position = light.transform.position, rotation = light.transform.rotation, color = light.color,
            intensity = light.intensity, range = light.range, enabled = light.enabled,
            inner = light.innerSpotAngle, outer = light.spotAngle, active = light.gameObject.activeInHierarchy })));
    }
    [Serializable] class LightState
    {
        public Vector3 position; public Quaternion rotation; public Color color;
        public float intensity, range, inner, outer; public bool enabled, active;
    }
}
