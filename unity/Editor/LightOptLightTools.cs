// Install in Assets/Editor. Changes are limited to eligible lights in the active scene.
using System;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace LightOpt.Reproduction
{
    public static class LightOptLightTools
    {
        [MenuItem("Tools/LightOpt/Disable temperature and cookies")]
        public static void DisableTemperatureAndCookies()
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode)
                throw new InvalidOperationException("Stop Play Mode before preparing lights.");

            var scene = SceneManager.GetActiveScene();
            if (!scene.IsValid() || !scene.isLoaded)
                throw new InvalidOperationException("No active scene is loaded.");

            // Match the default sampling selection, without changing light activation or mode.
            var lights = scene.GetRootGameObjects()
                .SelectMany(root => root.GetComponentsInChildren<Light>(true))
                .Where(light => light.enabled && light.gameObject.activeInHierarchy
                    && light.lightmapBakeType == LightmapBakeType.Realtime
                    && (light.type == LightType.Point || light.type == LightType.Spot)
                    && light.intensity > 0 && light.range > 0)
                .ToArray();
            var changed = lights.Where(light => light.useColorTemperature || light.cookie != null).ToArray();
            if (changed.Length == 0)
            {
                Debug.Log($"LightOpt: {lights.Length} eligible lights; no temperature or cookies to disable.");
                return;
            }

            int temperatureCount = changed.Count(light => light.useColorTemperature);
            int cookieCount = changed.Count(light => light.cookie != null);
            Undo.IncrementCurrentGroup();
            int undoGroup = Undo.GetCurrentGroup();
            Undo.SetCurrentGroupName("LightOpt: disable temperature and cookies");

            foreach (var light in changed)
            {
                Undo.RecordObject(light, "Prepare LightOpt light");
                light.useColorTemperature = false;
                light.cookie = null;
                // Record scene-instance overrides rather than editing prefab assets.
                if (PrefabUtility.IsPartOfPrefabInstance(light))
                    PrefabUtility.RecordPrefabInstancePropertyModifications(light);
                EditorUtility.SetDirty(light);
            }
            Undo.CollapseUndoOperations(undoGroup);
            EditorSceneManager.MarkSceneDirty(scene);
            Debug.Log($"LightOpt: prepared {changed.Length}/{lights.Length} eligible lights in '{scene.name}'. " +
                $"Disabled temperature on {temperatureCount}, removed cookies from {cookieCount}. " +
                "Save the scene and export to a new folder. Ctrl+Z undoes this operation.");
        }
    }
}
