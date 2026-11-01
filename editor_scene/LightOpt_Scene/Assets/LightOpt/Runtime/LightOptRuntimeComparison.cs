using System;
using System.Linq;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.Rendering;

namespace LightOpt.Reproduction
{
    [DisallowMultipleComponent]
    public sealed class LightOptRuntimeComparison : MonoBehaviour
    {
        [Serializable] public class State
        {
            public bool enabled = true;
            public float intensity, range;
            public Vector3 position;
        }
        [Serializable] public class Entry
        {
            public Light light;
            public State before, after, reference;
        }
        [Serializable] public class LossSnapshot
        {
            public double reconstruction, overlap, attraction, total;
        }
        [SerializeField] Entry[] entries = Array.Empty<Entry>();
        [SerializeField] int initialState;
        [SerializeField] LossSnapshot beforeLoss, afterLoss;
        [SerializeField] bool hasRecordedLosses;
        int currentState;
        string error;
        [SerializeField] string adjustmentSummary;
        Camera navigationCamera;
        static readonly string[] Names = { "Before", "After", "Reference" };

        public void Configure(Entry[] lights, int state, LossSnapshot initialLoss = null, LossSnapshot finalLoss = null, string changes = null)
        {
            if (lights == null || lights.Length == 0 || state < 0 || state > 2)
                throw new ArgumentException("Invalid runtime comparison configuration");
            entries = lights.OrderBy(entry => Depth(entry.light.transform)).ToArray();
            initialState = state;
            adjustmentSummary = changes;
            beforeLoss = initialLoss;
            afterLoss = finalLoss;
            hasRecordedLosses = initialLoss != null && finalLoss != null;
        }

        static int Depth(Transform node)
        {
            int depth = 0;
            while (node.parent != null) { ++depth; node = node.parent; }
            return depth;
        }

        void Start()
        {
            navigationCamera = GetComponent<Camera>();
            ApplyState(initialState);
            // Subscribe after the sample manager's initialization so stale preview copies also work.
            RenderPipelineManager.beginCameraRendering += BeforeCameraRendering;
        }

        void OnDisable() => RenderPipelineManager.beginCameraRendering -= BeforeCameraRendering;

        void BeforeCameraRendering(ScriptableRenderContext context, Camera camera)
        {
            if (camera == navigationCamera) ApplyState(currentState);
        }

        void Update()
        {
            var keyboard = Keyboard.current;
            if (!Application.isFocused || keyboard == null) return;
            if (keyboard.digit1Key.wasPressedThisFrame || keyboard.numpad1Key.wasPressedThisFrame) ApplyState(0);
            if (keyboard.digit2Key.wasPressedThisFrame || keyboard.numpad2Key.wasPressedThisFrame) ApplyState(1);
            if (keyboard.digit3Key.wasPressedThisFrame || keyboard.numpad3Key.wasPressedThisFrame) ApplyState(2);
        }

        public void ApplyState(int state)
        {
            if (state < 0 || state > 2) throw new ArgumentOutOfRangeException(nameof(state));
            if (entries.Length == 0 || entries.Any(entry => entry == null || entry.light == null || entry.before == null || entry.after == null || entry.reference == null))
            {
                error = "Missing comparison lights. Stop Play and reopen the preview panel.";
                return;
            }
            // Only light properties change. The navigation camera pose is left untouched.
            foreach (var entry in entries)
            {
                State selected = state == 0 ? entry.before : state == 1 ? entry.after : entry.reference;
                entry.light.enabled = selected.enabled;
                entry.light.intensity = selected.intensity;
                entry.light.range = selected.range;
                entry.light.transform.position = selected.position;
            }
            currentState = state;
            error = null;
        }

        void OnGUI()
        {
            GUILayout.BeginArea(new Rect(12, 74, 760, 300), GUI.skin.box);
            GUILayout.Label("Lighting: " + Names[currentState] + "  (1 / 2 / 3)");
            GUILayout.BeginHorizontal();
            for (int i = 0; i < Names.Length; ++i)
                if (GUILayout.Button((i + 1) + " " + Names[i], GUILayout.Height(30))) ApplyState(i);
            GUILayout.EndHorizontal();
            GUILayout.Label(error ?? "Release RMB to click buttons. Switching keeps the current camera view.");
            if (entries.Length > 0 && entries.All(entry => entry != null && entry.light != null))
            {
                int active = entries.Count(entry => entry.light.enabled && entry.light.gameObject.activeInHierarchy);
                int matching = entries.Count(entry => MatchesCurrent(entry));
                GUILayout.Label("Compared lights: " + entries.Length + " / Enabled: " + active + " / Matching: " + matching + " / Intensity sum: " + entries.Sum(entry => entry.light.intensity).ToString("F2"));
            }
            if (!string.IsNullOrEmpty(adjustmentSummary)) GUILayout.Label("Light set: " + adjustmentSummary);
            GUILayout.Space(5);
            GUILayout.Label("Saved Python losses for selected state (not live Unity metrics)");
            var losses = !hasRecordedLosses || currentState == 2 ? null : currentState == 0 ? beforeLoss : afterLoss;
            GUILayout.Label("Reconstruction: " + (losses == null ? "N/A" : losses.reconstruction.ToString("G6")));
            GUILayout.Label("Overlap: " + (losses == null ? "N/A" : losses.overlap.ToString("G6")));
            GUILayout.Label("Attraction: " + (losses == null ? "N/A" : losses.attraction.ToString("G6")));
            GUILayout.Label("Total: " + (losses == null ? "N/A" : losses.total.ToString("G6")));
            if (currentState == 2) GUILayout.Label("Reference losses were not recorded.");
            GUILayout.EndArea();
        }

        bool MatchesCurrent(Entry entry)
        {
            State selected = currentState == 0 ? entry.before : currentState == 1 ? entry.after : entry.reference;
            return selected != null && entry.light.enabled == selected.enabled && Mathf.Abs(entry.light.intensity-selected.intensity) < 1e-4f &&
                Mathf.Abs(entry.light.range-selected.range) < 1e-4f &&
                (entry.light.transform.position-selected.position).magnitude < 1e-4f;
        }
    }
}
