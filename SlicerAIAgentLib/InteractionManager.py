"""
InteractionManager - Low-level Slicer 3D interaction management.

Handles markup node creation, placement mode entry/exit, VTK observer
management with debounce timers, and completion detection for guided
interactive workflows.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import slicer
import qt
import vtk

logger = logging.getLogger(__name__)

# VTK event constants (numeric IDs for common markup events)
POINT_MODIFIED_EVENT = 70  # vtkMRMLMarkupsNode::PointModifiedEvent
POINT_ADDED_EVENT = 71     # vtkMRMLMarkupsNode::PointPositionDefinedEvent
INTERACTION_EVENT = 33     # vtkCommand::InteractionEvent


class InteractionManager:
    """
    Manages Slicer 3D view interactions for guided workflows.

    Responsibilities:
    - Creating markup nodes and entering placement modes
    - Setting up VTK observers with debounced callbacks
    - Detecting when user finishes interaction
    - Cleaning up observers and timers
    """

    # Map interaction_type strings to MRML node classes
    NODE_CLASS_MAP = {
        "curve": "vtkMRMLMarkupsCurveNode",
        "plane": "vtkMRMLMarkupsPlaneNode",
        "line": "vtkMRMLMarkupsLineNode",
        "fiducial": "vtkMRMLMarkupsFiducialNode",
        "point_list": "vtkMRMLMarkupsFiducialNode",
    }

    # Placement mode constants
    PLACE_INDEFINITE = "indefinite"  # Keep placing until user stops
    PLACE_SINGLE = "single"          # One point/interaction then stop

    def __init__(self):
        self._observers: Dict[str, List[int]] = {}  # node_id -> [observer_tags]
        self._debounce_timers: Dict[str, qt.QTimer] = {}  # key -> QTimer
        self._pending_debounce_callbacks: Dict[str, Callable] = {}
        self._placement_active: bool = False
        self._completion_callback: Optional[Callable] = None
        self._active_placement_node_id: Optional[str] = None
        self._all_created_node_ids: List[str] = []

    def create_and_place(
        self,
        interaction_type: str,
        name: str,
        seed_points: Optional[List[List[float]]] = None,
        placement_mode: str = "indefinite",
        completion_callback: Optional[Callable] = None,
        display_properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a markup node, optionally add seed points, and enter Slicer's
        placement mode.

        Args:
            interaction_type: One of "curve", "plane", "line", "fiducial", "point_list".
            name: Display name for the created node.
            seed_points: Optional list of [x, y, z] initial control points.
            placement_mode: "indefinite" (keep placing) or "single" (one point).
            completion_callback: Called when placement completes (not via this method,
                but stored for external triggering).
            display_properties: Optional dict of display node properties
                (e.g., {"color": [1,0,0], "visibility": True}).

        Returns:
            Node ID of the created markup node.
        """
        node_class = self.NODE_CLASS_MAP.get(interaction_type)
        if not node_class:
            raise ValueError(
                f"Unknown interaction_type '{interaction_type}'. "
                f"Must be one of: {list(self.NODE_CLASS_MAP.keys())}"
            )

        # Create the node
        node = slicer.mrmlScene.AddNewNodeByClass(node_class, name)
        if node is None:
            raise RuntimeError(f"Failed to create {node_class} named '{name}'")

        node_id = node.GetID()
        self._active_placement_node_id = node_id
        self._all_created_node_ids.append(node_id)
        self._completion_callback = completion_callback

        # Add seed points if provided
        if seed_points:
            self.add_seed_points(node, seed_points)

        # Set up display properties
        display_node = node.GetDisplayNode()
        if display_node is None:
            display_node = node.CreateDefaultDisplayNode()
        if display_properties:
            self._apply_display_properties(display_node, display_properties)
        display_node.SetVisibility(True)

        # Enter placement mode
        self._enter_placement_mode(node, placement_mode)
        self._placement_active = True

        logger.info(
            f"[InteractionManager] Created {interaction_type} node '{name}' "
            f"(ID: {node_id}), placement mode active"
        )
        return node_id

    def add_seed_points(self, node, points: List[List[float]]) -> None:
        """Add initial control points to a markup node."""
        for pt in points:
            if len(pt) >= 3:
                node.AddControlPoint(vtk.vtkVector3d(pt[0], pt[1], pt[2]))

    def setup_debounced_observer(
        self,
        node_id: str,
        event_type: int,
        callback: Callable,
        debounce_ms: int = 300,
        observer_key: Optional[str] = None,
    ) -> int:
        """
        Add a debounced VTK observer to a node.

        The callback is not called on every event. Instead, it fires after
        `debounce_ms` milliseconds of silence following the last event.

        Args:
            node_id: MRML node ID to observe.
            event_type: VTK event type constant (e.g., POINT_MODIFIED_EVENT).
            callback: Function to call after debounce period.
            debounce_ms: Debounce interval in milliseconds.
            observer_key: Optional key for timer tracking. Defaults to node_id.

        Returns:
            VTK observer tag.
        """
        node = slicer.mrmlScene.GetNodeByID(node_id)
        if node is None:
            raise ValueError(f"Node not found: {node_id}")

        key = observer_key or f"{node_id}_{event_type}"

        # Create or reset debounce timer
        if key in self._debounce_timers:
            self._debounce_timers[key].stop()
        else:
            self._debounce_timers[key] = qt.QTimer()
            self._debounce_timers[key].singleShot = True

        self._pending_debounce_callbacks[key] = callback

        def _on_debounce_fire():
            cb = self._pending_debounce_callbacks.get(key)
            if cb:
                try:
                    cb()
                except Exception as e:
                    logger.error(f"[InteractionManager] Debounced callback error: {e}")

        self._debounce_timers[key].timeout.connect(_on_debounce_fire)

        # VTK observer that resets the debounce timer on each event
        def _on_vtk_event(caller, event):
            timer = self._debounce_timers.get(key)
            if timer:
                timer.start(debounce_ms)
            return 0

        tag = node.AddObserver(event_type, _on_vtk_event)

        # Track observer for cleanup
        if node_id not in self._observers:
            self._observers[node_id] = []
        self._observers[node_id].append(tag)

        logger.info(
            f"[InteractionManager] Added debounced observer on {node_id} "
            f"event={event_type} debounce={debounce_ms}ms"
        )
        return tag

    def setup_observer(
        self,
        node_id: str,
        event_type: int,
        callback: Callable,
    ) -> int:
        """
        Add a direct (non-debounced) VTK observer to a node.

        Args:
            node_id: MRML node ID to observe.
            event_type: VTK event type constant.
            callback: Function(caller, event) to call on event.

        Returns:
            VTK observer tag.
        """
        node = slicer.mrmlScene.GetNodeByID(node_id)
        if node is None:
            raise ValueError(f"Node not found: {node_id}")

        tag = node.AddObserver(event_type, callback)

        if node_id not in self._observers:
            self._observers[node_id] = []
        self._observers[node_id].append(tag)

        return tag

    def exit_placement_mode(self) -> None:
        """Exit Slicer's placement mode and return to view transform mode."""
        try:
            interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
            if interactionNode:
                interactionNode.SwitchToViewTransformMode()
        except Exception as e:
            logger.warning(f"[InteractionManager] Failed to exit placement mode: {e}")
        self._placement_active = False

    def is_placement_active(self) -> bool:
        """Check if placement mode is currently active."""
        return self._placement_active

    def get_node_control_point_count(self, node_id: str) -> int:
        """Get the number of control points on a markup node."""
        node = slicer.mrmlScene.GetNodeByID(node_id)
        if node is None:
            return 0
        return node.GetNumberOfControlPoints()

    def validate_node(self, node_id: str, min_points: int = 0) -> Tuple[bool, str]:
        """
        Validate a markup node's state after user interaction.

        Returns:
            Tuple of (is_valid, error_message).
        """
        node = slicer.mrmlScene.GetNodeByID(node_id)
        if node is None:
            return False, f"Node not found: {node_id}"

        n_points = node.GetNumberOfControlPoints()
        if min_points > 0 and n_points < min_points:
            return False, (
                f"Node '{node.GetName()}' needs at least {min_points} "
                f"control points, got {n_points}. Please add more points."
            )
        return True, ""

    def cleanup(self, node_id: Optional[str] = None) -> None:
        """
        Remove observers and timers. If node_id is None, clean up everything.
        """
        if node_id:
            self._cleanup_node(node_id)
        else:
            for nid in list(self._observers.keys()):
                self._cleanup_node(nid)
            for key in list(self._debounce_timers.keys()):
                self._debounce_timers[key].stop()
            self._debounce_timers.clear()
            self._pending_debounce_callbacks.clear()
            self._observers.clear()
            self._placement_active = False
            self._active_placement_node_id = None

    def cleanup_all_created_nodes(self) -> None:
        """Remove all markup nodes created by this manager from the scene."""
        for node_id in self._all_created_node_ids:
            node = slicer.mrmlScene.GetNodeByID(node_id)
            if node:
                slicer.mrmlScene.RemoveNode(node)
        self._all_created_node_ids.clear()

    def get_active_placement_node_id(self) -> Optional[str]:
        """Return the node ID of the currently active placement node."""
        return self._active_placement_node_id

    # --- Private methods ---

    def _enter_placement_mode(self, node, mode: str) -> None:
        """Enter Slicer's markup placement mode."""
        try:
            markupsLogic = slicer.modules.markups.logic()
            markupsLogic.SetActiveListID(node)
            interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")

            if mode == self.PLACE_SINGLE:
                interactionNode.SwitchToSinglePlaceMode()
            else:
                interactionNode.SwitchToPersistentPlaceMode()
        except Exception as e:
            logger.error(f"[InteractionManager] Failed to enter placement mode: {e}")
            raise

    def _apply_display_properties(self, display_node, props: Dict[str, Any]) -> None:
        """Apply display property dict to a markup display node."""
        if "color" in props:
            c = props["color"]
            display_node.SetColor(c[0], c[1], c[2])
        if "selectedColor" in props:
            c = props["selectedColor"]
            display_node.SetSelectedColor(c[0], c[1], c[2])
        if "opacity" in props:
            display_node.SetOpacity(props["opacity"])
        if "pointSize" in props:
            display_node.SetPointSize(props["pointSize"])
        if "lineWidth" in props:
            display_node.SetLineWidth(props["lineWidth"])

        # Interaction handle visibility
        if props.get("rotationHandles"):
            display_node.RotationHandleVisibilityOn()
        else:
            display_node.RotationHandleVisibilityOff()
        if props.get("translationHandles"):
            display_node.TranslationHandleVisibilityOn()
        else:
            display_node.TranslationHandleVisibilityOff()
        if props.get("scaleHandles", False) is False:
            display_node.ScaleHandleVisibilityOff()

    def _cleanup_node(self, node_id: str) -> None:
        """Remove observers and timers for a specific node."""
        node = slicer.mrmlScene.GetNodeByID(node_id)

        # Remove VTK observers
        tags = self._observers.pop(node_id, [])
        if node:
            for tag in tags:
                try:
                    node.RemoveObserver(tag)
                except Exception:
                    pass

        # Remove debounce timers associated with this node
        keys_to_remove = [
            k for k in self._debounce_timers
            if k.startswith(node_id)
        ]
        for key in keys_to_remove:
            self._debounce_timers[key].stop()
            del self._debounce_timers[key]
            self._pending_debounce_callbacks.pop(key, None)
