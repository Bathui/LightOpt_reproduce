using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.Rendering;
using UnityEditor.Rendering.Universal;
using UnityEngine;

public static class LightOptCourtyardMaterials
{
    const string Root = "Assets/Unity Technologies/The Courtyard";
    const string Lit = "Universal Render Pipeline/Lit";

    [Serializable]
    public class Entry
    {
        public string path, beforeShader, afterShader, status, note;
        public bool supported, baseTexturePreserved, baseColorPreserved;
    }

    [Serializable]
    public class Report
    {
        public string backupDirectory;
        public int converted, skipped, failed;
        public List<Entry> materials = new List<Entry>();
    }

    [MenuItem("Tools/LightOpt/Convert Courtyard Materials to URP")]
    public static void ConvertMaterials()
    {
        if (EditorApplication.isPlayingOrWillChangePlaymode)
            throw new InvalidOperationException("Stop Play Mode before converting materials.");
        if (Shader.Find(Lit) == null)
            throw new InvalidOperationException("URP Lit shader is unavailable.");

        string project = Directory.GetParent(Application.dataPath).FullName;
        string audit = Path.GetFullPath(Path.Combine(project, "../../tmp/review/courtyard-materials"));
        string backup = Path.Combine(audit, DateTime.UtcNow.ToString("yyyyMMdd-HHmmss-fff"));
        Directory.CreateDirectory(backup);
        var report = new Report { backupDirectory = backup };
        string[] guids = AssetDatabase.FindAssets("t:Material", new[] { Root });
        Array.Sort(guids, StringComparer.Ordinal);
        Undo.IncrementCurrentGroup();
        int undoGroup = Undo.GetCurrentGroup();
        Undo.SetCurrentGroupName("Convert Courtyard Materials to URP");
        try
        {
            foreach (string guid in guids)
            {
                string path = AssetDatabase.GUIDToAssetPath(guid);
                if (!path.StartsWith(Root + "/", StringComparison.Ordinal) || !path.EndsWith(".mat"))
                    continue;
                var material = AssetDatabase.LoadAssetAtPath<Material>(path);
                if (material == null) continue;
                string shader = material.shader != null ? material.shader.name : "<missing>";
                var entry = new Entry { path = path, beforeShader = shader };
                report.materials.Add(entry);
                bool standard = shader == "Standard" || shader == "Standard (Specular setup)" ||
                    shader == "Standard DirSpecHalfLambert" || shader == "Standard DirSpecHalfLambert (Specular setup)" ||
                    shader == "Nature/Tree Creator Bark" || shader == "Autodesk Interactive";
                bool water = shader == "FX/Water4";
                bool shadowOnly = shader == "Standard - Shadows Only";
                bool diffuse = shader == "Legacy Shaders/Diffuse" || shader == "Diffuse";
                bool missing = shader == "<missing>" || shader == "Hidden/InternalErrorShader";
                if (!standard && !diffuse && !missing && !water && !shadowOnly)
                {
                    entry.afterShader = shader;
                    entry.status = "skipped";
                    entry.note = shader.StartsWith("Universal Render Pipeline/") ? "Already URP." :
                        "Special shader retained; needs separate visual review.";
                    entry.supported = material.shader != null && material.shader.isSupported;
                    report.skipped++;
                    continue;
                }

                string backupPath = Path.Combine(backup, path);
                Directory.CreateDirectory(Path.GetDirectoryName(backupPath));
                File.Copy(path, backupPath, false);
                File.Copy(path + ".meta", backupPath + ".meta", false);
                var original = new Material(material);
                try
                {
                    Undo.RecordObject(material, "Convert Courtyard Material");
                    // A missing legacy shader can still retain serialized Standard properties.
                    if (missing) material.shader = Shader.Find("Standard");
                    Texture baseTexture = material.HasProperty("_MainTex") ? material.GetTexture("_MainTex") : null;
                    Color baseColor = material.HasProperty("_Color") ? material.GetColor("_Color") : Color.white;
                    Vector2 scale = material.HasProperty("_MainTex") ? material.GetTextureScale("_MainTex") : Vector2.one;
                    Vector2 offset = material.HasProperty("_MainTex") ? material.GetTextureOffset("_MainTex") : Vector2.zero;
                    if (shadowOnly)
                    {
                        Shader replacement = Shader.Find("LightOpt/Courtyard Shadows Only URP");
                        if (replacement == null) throw new InvalidOperationException("URP shadow-only shader is unavailable.");
                        material.shader = replacement;
                        material.shaderKeywords = Array.Empty<string>();
                        material.renderQueue = 2000;
                    }
                    else if (water)
                    {
                        Color tint = material.HasProperty("_DepthColor") ? material.GetColor("_DepthColor") : new Color(.05f, .2f, .2f, .5f);
                        material.shader = Shader.Find(Lit);
                        material.shaderKeywords = Array.Empty<string>();
                        material.SetTexture("_BaseMap", null);
                        material.SetColor("_BaseColor", tint);
                        material.SetFloat("_Metallic", 0);
                        material.SetFloat("_Smoothness", .9f);
                        material.SetFloat("_Surface", 1);
                        material.SetFloat("_Blend", 0);
                        material.SetFloat("_AlphaClip", 0);
                        baseTexture = null;
                        baseColor = tint;
                        material.SetTextureScale("_BaseMap", scale);
                        material.SetTextureOffset("_BaseMap", offset);
                    }
                    else if (diffuse)
                    {
                        material.shader = Shader.Find(Lit);
                        material.SetTexture("_BaseMap", baseTexture);
                        material.SetTextureScale("_BaseMap", scale);
                        material.SetTextureOffset("_BaseMap", offset);
                        material.SetColor("_BaseColor", baseColor);
                        material.SetFloat("_Metallic", 0);
                        material.SetFloat("_Smoothness", 0);
                        material.shaderKeywords = Array.Empty<string>();
                    }
                    else
                    {
                        new StandardUpgrader(material.shader.name).Upgrade(material, MaterialUpgrader.UpgradeFlags.None);
                    }
                    // Validate keywords with the installed URP shader GUI, including normal/emission maps.
                    if (!shadowOnly)
                    {
                        Type guiType = typeof(BaseShaderGUI).Assembly.GetType("UnityEditor.Rendering.Universal.ShaderGUI.LitShader");
                        ((ShaderGUI)Activator.CreateInstance(guiType, true)).ValidateMaterial(material);
                    }
                    entry.baseTexturePreserved = shadowOnly || (material.GetTexture("_BaseMap") == baseTexture &&
                        material.GetTextureScale("_BaseMap") == scale && material.GetTextureOffset("_BaseMap") == offset);
                    entry.baseColorPreserved = shadowOnly || material.GetColor("_BaseColor") == baseColor;
                    if ((!shadowOnly && material.shader.name != Lit) || !entry.baseTexturePreserved || !entry.baseColorPreserved)
                        throw new InvalidOperationException("Material property preservation check failed.");
                    entry.afterShader = material.shader.name;
                    entry.supported = material.shader.isSupported;
                    entry.status = "converted";
                    entry.note = shadowOnly ? "URP shadow caster with invisible color pass; no opaque replacement surface." :
                        water ? "Static transparent URP Lit approximation using original depth tint; legacy waves/refraction/GrabPass are not reproduced." :
                        shader.Contains("HalfLambert") ? "URP Lit approximation; custom half-Lambert response is not reproduced." :
                        missing ? "Recovered serialized Standard properties from a missing shader." : "Built-in properties migrated to URP Lit.";
                    EditorUtility.SetDirty(material);
                    AssetDatabase.SaveAssetIfDirty(material);
                    report.converted++;
                }
                catch (Exception error)
                {
                    material.shader = original.shader;
                    material.CopyPropertiesFromMaterial(original);
                    entry.status = "failed";
                    entry.note = error.ToString();
                    report.failed++;
                    Debug.LogException(error);
                }
                finally { UnityEngine.Object.DestroyImmediate(original); }
            }
        }
        finally
        {
            Undo.CollapseUndoOperations(undoGroup);
            File.WriteAllText(Path.Combine(backup, "report.json"), JsonUtility.ToJson(report, true));
            File.WriteAllText(Path.Combine(audit, "latest-report.json"), JsonUtility.ToJson(report, true));
            SceneView.RepaintAll();
        }
        Debug.Log($"Courtyard URP conversion: {report.converted} converted, {report.skipped} retained, {report.failed} failed. Backup: {backup}");
    }
}
