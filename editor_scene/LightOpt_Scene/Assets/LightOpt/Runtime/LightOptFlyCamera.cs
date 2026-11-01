using UnityEngine;
using UnityEngine.InputSystem;

namespace LightOpt.Reproduction
{
    [DisallowMultipleComponent]
    [RequireComponent(typeof(Camera))]
    [DefaultExecutionOrder(10000)]
    public sealed class LightOptFlyCamera : MonoBehaviour
    {
        public float moveSpeed = 8f;
        public float fastMultiplier = 4f;
        public float mouseSensitivity = .12f;
        Vector3 startPosition;
        Quaternion startRotation;
        float yaw, pitch;
        bool looking;

        void Start()
        {
            startPosition = transform.position;
            startRotation = transform.rotation;
            ReadAngles();
        }

        void ReadAngles()
        {
            yaw = transform.eulerAngles.y;
            pitch = Mathf.DeltaAngle(0, transform.eulerAngles.x);
        }

        void LateUpdate()
        {
            var mouse = Mouse.current;
            var keyboard = Keyboard.current;
            if (!Application.isFocused || mouse == null || keyboard == null)
            {
                ReleaseCursor();
                return;
            }
            if (keyboard.escapeKey.wasPressedThisFrame || mouse.rightButton.wasReleasedThisFrame)
                ReleaseCursor();
            if (mouse.rightButton.wasPressedThisFrame)
            {
                ReadAngles();
                looking = true;
                Cursor.lockState = CursorLockMode.Locked;
                Cursor.visible = false;
            }
            if (!looking)
            {
                Cursor.lockState = CursorLockMode.None;
                Cursor.visible = true;
                return;
            }

            Vector2 delta = mouse.delta.ReadValue();
            yaw += delta.x * mouseSensitivity;
            pitch = Mathf.Clamp(pitch - delta.y * mouseSensitivity, -89f, 89f);
            transform.rotation = Quaternion.Euler(pitch, yaw, 0);
            Vector3 move = Vector3.zero;
            if (keyboard.wKey.isPressed) move += transform.forward;
            if (keyboard.sKey.isPressed) move -= transform.forward;
            if (keyboard.dKey.isPressed) move += transform.right;
            if (keyboard.aKey.isPressed) move -= transform.right;
            if (keyboard.eKey.isPressed) move += Vector3.up;
            if (keyboard.qKey.isPressed) move -= Vector3.up;
            float speed = moveSpeed * (keyboard.leftShiftKey.isPressed || keyboard.rightShiftKey.isPressed ? fastMultiplier : 1f);
            transform.position += Vector3.ClampMagnitude(move, 1f) * speed * Mathf.Min(Time.unscaledDeltaTime, .1f);
            moveSpeed = Mathf.Clamp(moveSpeed * Mathf.Exp(mouse.scroll.ReadValue().y * .001f), .2f, 200f);
            if (keyboard.rKey.wasPressedThisFrame)
            {
                transform.SetPositionAndRotation(startPosition, startRotation);
                ReadAngles();
            }
        }

        void ReleaseCursor()
        {
            if (!looking) return;
            looking = false;
            Cursor.lockState = CursorLockMode.None;
            Cursor.visible = true;
        }

        void OnDisable() => ReleaseCursor();
        void OnApplicationFocus(bool focused) { if (!focused) ReleaseCursor(); }

        void OnGUI()
        {
            GUI.Box(new Rect(12, 12, 560, 54),
                "Hold RMB: look + WASD move | Q/E: down/up | Shift: faster\n" +
                "Wheel: speed (" + moveSpeed.ToString("F1") + ") | R: reset while RMB held | Esc: release mouse");
        }
    }
}
