using UnityEngine;

public class FlashLight : MonoBehaviour {
	public void SetLight(bool set) {
		if (!this || !isActiveAndEnabled) return;
		GetComponent<Light>().enabled = set;
	}
}
