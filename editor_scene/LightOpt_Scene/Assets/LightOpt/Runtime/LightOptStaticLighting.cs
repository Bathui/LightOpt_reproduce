using UnityEngine;

// Stop legacy scene drivers without overwriting the lights used by Before/After comparison.
[ExecuteAlways, DefaultExecutionOrder(-32000), DisallowMultipleComponent]
public sealed class LightOptStaticLighting : MonoBehaviour
{
    public Behaviour[] controllers = new Behaviour[0];
    public GameObject[] timeControls = new GameObject[0];
    public Rigidbody[] bodies = new Rigidbody[0];

    void Awake() { Enforce(); }
    void OnEnable() { Enforce(); }
    void Update() { Enforce(); }

    public void Enforce()
    {
        foreach (Rigidbody body in bodies)
            if (body != null && !body.isKinematic) body.isKinematic = true;
        foreach (Behaviour controller in controllers)
            if (controller != null && controller.enabled) controller.enabled = false;
        foreach (GameObject control in timeControls)
            if (control != null && control.activeSelf) control.SetActive(false);
    }
}
