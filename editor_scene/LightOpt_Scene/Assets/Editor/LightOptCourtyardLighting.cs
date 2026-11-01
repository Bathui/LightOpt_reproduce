using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.SceneManagement;

public static class LightOptCourtyardLighting
{
    [MenuItem("Tools/LightOpt/Create Courtyard URP Lighting Preview")]
    public static void CreatePreview()
    {
        if (EditorApplication.isPlayingOrWillChangePlaymode)
            throw new InvalidOperationException("Stop Play Mode first.");
        var source = SceneManager.GetActiveScene();
        if (source.path != "Assets/Unity Technologies/The Courtyard/Scene/Courtyard.unity")
            throw new InvalidOperationException("Open the original Courtyard scene first.");
        string target = "Assets/Scenes/Courtyard_LightOpt.unity";
        if (File.Exists(target))
            throw new InvalidOperationException("Courtyard_LightOpt already exists; it was not overwritten.");
        Inspect();
        string project = Directory.GetParent(Application.dataPath).FullName;
        string audit = Path.GetFullPath(Path.Combine(project, "../../tmp/review/courtyard-lighting"));
        File.Copy(Path.Combine(audit, "lighting-report.json"), Path.Combine(audit, "before.json"), true);
        if (!EditorSceneManager.SaveScene(source, target, true))
            throw new IOException("Unable to create the experimental scene copy.");
        var scene = EditorSceneManager.OpenScene(target, OpenSceneMode.Single);
        // Freeze only the legacy environment/exposure drivers in the experiment copy.
        foreach (GameObject root in scene.GetRootGameObjects())
            foreach (MonoBehaviour behaviour in root.GetComponentsInChildren<MonoBehaviour>(true))
            {
                if (behaviour == null) continue;
                string type = behaviour.GetType().Name;
                if (type == "TimeOfDayManager" || type == "SunProperties" ||
                    type == "EnvironmentUpdater" || type == "ColorGradingProperties")
                    behaviour.enabled = false;
            }
        string assetFolder = "Assets/LightOpt/Courtyard";
        if (!AssetDatabase.IsValidFolder(assetFolder)) AssetDatabase.CreateFolder("Assets/LightOpt", "Courtyard");
        var sky = new Material(RenderSettings.skybox);
        sky.name = "Courtyard_URP_Sky";
        sky.SetFloat("_Exposure", 1);
        string skyPath = AssetDatabase.GenerateUniqueAssetPath(assetFolder + "/Courtyard_URP_Sky.mat");
        AssetDatabase.CreateAsset(sky, skyPath);
        RenderSettings.skybox = sky;
        RenderSettings.ambientMode = AmbientMode.Flat;
        RenderSettings.ambientLight = new Color(.15f, .15f, .15f);
        RenderSettings.ambientIntensity = 1;

        var profile = ScriptableObject.CreateInstance<VolumeProfile>();
        profile.name = "Courtyard_URP_Exposure";
        string profilePath = AssetDatabase.GenerateUniqueAssetPath(assetFolder + "/Courtyard_URP_Exposure.asset");
        AssetDatabase.CreateAsset(profile, profilePath);
        var tone = profile.Add<Tonemapping>(true);
        tone.mode.Override(TonemappingMode.ACES);
        var color = profile.Add<ColorAdjustments>(true);
        color.postExposure.Override(-1.5f);
        foreach (VolumeComponent component in profile.components)
            AssetDatabase.AddObjectToAsset(component, profile);
        var volumeObject = new GameObject("__LightOpt_URP_Exposure");
        var volume = volumeObject.AddComponent<Volume>();
        volume.isGlobal = true;
        volume.priority = 1000;
        volume.sharedProfile = profile;
        foreach (GameObject root in scene.GetRootGameObjects())
            foreach (Camera camera in root.GetComponentsInChildren<Camera>(true))
            {
                var data = camera.GetUniversalAdditionalCameraData();
                data.renderPostProcessing = true;
                data.volumeLayerMask |= 1;
            }
        AssetDatabase.SaveAssetIfDirty(profile);
        LightOptCourtyardStaticLighting.Freeze(scene);
        EditorSceneManager.MarkSceneDirty(scene);
        if (!EditorSceneManager.SaveScene(scene)) throw new IOException("Unable to save experimental scene.");
        DynamicGI.UpdateEnvironment();
        SceneView.RepaintAll();
        Inspect();
        File.Copy(Path.Combine(audit, "lighting-report.json"), Path.Combine(audit, "after.json"), true);
        Debug.Log("Created Courtyard_LightOpt: local light parameters preserved; ambient reduced; ACES and -1.5 EV enabled.");
    }
    [Serializable] public class LightInfo
    {
        public string name, type;
        public float intensity;
        public Color color;
    }
    [Serializable] public class LightingReport
    {
        public string scene, ambientMode, skyShader;
        public float ambientIntensity, skyExposure;
        public Color ambientLight;
        public List<LightInfo> lights = new List<LightInfo>();
    }
    [InitializeOnLoadMethod]
    static void ScheduleInspection() { EditorApplication.delayCall += Inspect; }

    [MenuItem("Tools/LightOpt/Inspect Courtyard Lighting")]
    public static void Inspect()
    {
        var scene = SceneManager.GetActiveScene();
        if (!scene.IsValid() || !scene.name.StartsWith("Courtyard")) return;
        var report = new LightingReport {
            scene = scene.path, ambientMode = RenderSettings.ambientMode.ToString(),
            ambientIntensity = RenderSettings.ambientIntensity, ambientLight = RenderSettings.ambientLight,
            skyShader = RenderSettings.skybox == null ? "<none>" : RenderSettings.skybox.shader.name,
            skyExposure = RenderSettings.skybox != null && RenderSettings.skybox.HasProperty("_Exposure") ?
                RenderSettings.skybox.GetFloat("_Exposure") : 0
        };
        foreach (GameObject root in scene.GetRootGameObjects())
            foreach (Light light in root.GetComponentsInChildren<Light>(true))
                if (light.enabled && light.gameObject.activeInHierarchy)
                    report.lights.Add(new LightInfo { name = light.name, type = light.type.ToString(),
                        intensity = light.intensity, color = light.color });
        report.lights.Sort((a, b) => b.intensity.CompareTo(a.intensity));
        string project = Directory.GetParent(Application.dataPath).FullName;
        string output = Path.GetFullPath(Path.Combine(project, "../../tmp/review/courtyard-lighting"));
        Directory.CreateDirectory(output);
        File.WriteAllText(Path.Combine(output, "lighting-report.json"), JsonUtility.ToJson(report, true));
    }
}
