import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox
import json
from pathlib import Path
from typing import Optional

from integra_pose.gui.scrollable import create_scrollable_section

class SkeletonEditor:
    def __init__(self, app):
        self.app = app
        self.root = tk.Toplevel(app.root)
        self.root.title("Skeleton Editor")
        self.root.geometry("1200x800")

        self.joints = []
        self.edges = []
        self.image_scale_x = 1.0
        self.image_scale_y = 1.0
        self._joint_reference_size: Optional[tuple[float, float]] = None
        self._active_joint_index: Optional[int] = None

        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Wrap the left-side controls column in a scrollable section so
        # the listboxes plus action buttons remain reachable on shorter
        # displays. The right-side canvas keeps its full height.
        left_column = ttk.Frame(self.main_frame, width=220)
        left_column.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_column.pack_propagate(False)
        _skel_canvas, left_panel = create_scrollable_section(self.root, left_column)

        ttk.Label(left_panel, text="Keypoints").pack(pady=5)
        self.keypoint_listbox = tk.Listbox(left_panel, selectmode=tk.SINGLE)
        self.keypoint_listbox.pack(fill=tk.BOTH, expand=True)
        self.keypoint_listbox.bind("<<ListboxSelect>>", self._on_joint_selected)

        keypoint_buttons_frame = ttk.Frame(left_panel)
        keypoint_buttons_frame.pack(fill=tk.X, pady=5)
        ttk.Button(keypoint_buttons_frame, text="Remove", command=self.remove_joint).pack(side=tk.LEFT, expand=True, fill=tk.X)

        ttk.Label(left_panel, text="Edges").pack(pady=5)
        self.edge_listbox = tk.Listbox(left_panel, selectmode=tk.SINGLE)
        self.edge_listbox.pack(fill=tk.BOTH, expand=True)

        edge_buttons_frame = ttk.Frame(left_panel)
        edge_buttons_frame.pack(fill=tk.X, pady=5)
        ttk.Button(edge_buttons_frame, text="Add Edge", command=self.add_edge).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(edge_buttons_frame, text="Remove Edge", command=self.remove_edge).pack(side=tk.LEFT, expand=True, fill=tk.X)

        ttk.Button(left_panel, text="Save Skeleton", command=self.save_skeleton).pack(fill=tk.X, pady=10)

        self.status_var = tk.StringVar(
            value="Tip: Select a joint, then click 'Add Edge' and another joint to connect them."
        )
        ttk.Label(
            left_panel,
            textvariable=self.status_var,
            wraplength=180,
            justify=tk.LEFT,
            foreground="#555555",
        ).pack(fill=tk.X, pady=(0, 8))

        right_panel = ttk.Frame(self.main_frame)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(right_panel, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)

        self.adding_edge = False
        self.edge_start_joint: Optional[int] = None

        self._load_initial_skeleton()
        self.update_keypoint_listbox(keep_selection=False)
        self.update_edge_listbox()

        self.status_var.set(
            "Tip: Click in the canvas to place joints. Use 'Add Edge' to connect joints."
        )

    def _load_initial_skeleton(self) -> None:
        names = self._get_keypoint_names()

        if not self.joints and names:
            self._create_default_layout(names)
        elif names:
            for idx, name in enumerate(names):
                if idx < len(self.joints):
                    if not self.joints[idx].get("name"):
                        self.joints[idx]["name"] = name
                else:
                    # Append missing joints so indices align with keypoint order.
                    ref_width, ref_height = self._joint_reference_size or (640.0, 480.0)
                    self.joints.append(
                        {
                            "name": name,
                            "x": float(ref_width) * 0.5,
                            "y": float(ref_height) * 0.5,
                        }
                    )

        if not self.edges:
            config_edges = getattr(self.app.config.pose_clustering, 'skeleton_connections', []) or []
            self.edges = [
                (int(edge[0]), int(edge[1]))
                for edge in config_edges
                if isinstance(edge, (list, tuple)) and len(edge) == 2
            ]

        if self.joints and self._joint_reference_size is None:
            # Default to a sensible drawing area if nothing else is known yet.
            self._joint_reference_size = (640.0, 480.0)

    def _get_keypoint_names(self) -> list[str]:
        if hasattr(self.app, 'keypoint_names') and self.app.keypoint_names:
            return list(self.app.keypoint_names)
        raw = self.app.config.setup.keypoint_names_str.get()
        if not raw:
            return []
        return [name.strip() for name in raw.split(',') if name.strip()]

    def _create_default_layout(self, names: list[str]) -> None:
        if not names:
            return
        width, height = 640.0, 480.0
        spacing = height / (len(names) + 1)
        self.joints = []
        for idx, name in enumerate(names):
            x = width * 0.5
            y = spacing * (idx + 1)
            self.joints.append({"name": name, "x": x, "y": y})
        self._joint_reference_size = (width, height)

    def _on_joint_selected(self, _event=None) -> None:
        selection = self.keypoint_listbox.curselection()
        self._active_joint_index = selection[0] if selection else None
        if self._active_joint_index is not None and self._active_joint_index < len(self.joints):
            name = self.joints[self._active_joint_index].get("name", f"kp_{self._active_joint_index}")
            self.status_var.set(f"Selected '{name}'. Click 'Add Edge' to start a connection.")
        else:
            self.status_var.set(
                "No joint selected. Click a joint or list entry to select it."
            )
        self.draw_skeleton()

    def _select_joint(self, index: int) -> None:
        if index < 0 or index >= len(self.joints):
            return
        self.keypoint_listbox.selection_clear(0, tk.END)
        self.keypoint_listbox.selection_set(index)
        self.keypoint_listbox.activate(index)
        self._active_joint_index = index
        name = self.joints[index].get("name", f"kp_{index}")
        self.status_var.set(f"Selected '{name}'. Click another joint to connect.")
        self.draw_skeleton()

    def on_canvas_click(self, event):
        if self.adding_edge:
            clicked_joint = self.get_joint_at(event.x, event.y)
            if clicked_joint:
                if self.edge_start_joint is None:
                    self.edge_start_joint = clicked_joint
                    self._select_joint(clicked_joint)
                    name = self.joints[clicked_joint].get("name", f"kp_{clicked_joint}")
                    self.status_var.set(f"Connecting from '{name}'. Click another joint.")
                else:
                    if clicked_joint != self.edge_start_joint:
                        candidate = (self.edge_start_joint, clicked_joint)
                        if candidate not in self.edges and (clicked_joint, self.edge_start_joint) not in self.edges:
                            self.edges.append(candidate)
                            name_a = self.joints[self.edge_start_joint].get("name", f"kp_{self.edge_start_joint}")
                            name_b = self.joints[clicked_joint].get("name", f"kp_{clicked_joint}")
                            self.status_var.set(f"Added edge {name_a} → {name_b}.")
                        else:
                            self.status_var.set("Edge already exists; no changes made.")
                    self.edge_start_joint = None
                    self.adding_edge = False
                    self.draw_skeleton()
                    self.update_edge_listbox()
            return
        else:
            clicked_joint = self.get_joint_at(event.x, event.y)
            if clicked_joint is not None:
                if (
                    self._active_joint_index is not None
                    and self._active_joint_index != clicked_joint
                ):
                    pair = (self._active_joint_index, clicked_joint)
                    reverse_pair = (clicked_joint, self._active_joint_index)
                    if pair not in self.edges and reverse_pair not in self.edges:
                        self.edges.append(pair)
                        self.update_edge_listbox()
                        name_a = self.joints[self._active_joint_index].get("name", f"kp_{self._active_joint_index}")
                        name_b = self.joints[clicked_joint].get("name", f"kp_{clicked_joint}")
                        self.status_var.set(f"Added edge {name_a} → {name_b}.")
                        self.draw_skeleton()
                    else:
                        self.status_var.set("Edge already exists; select different joints.")
                else:
                    self._select_joint(clicked_joint)
                return

            selection = self.keypoint_listbox.curselection()
            if selection:
                idx = selection[0]
                self.joints[idx]["x"] = event.x * self.image_scale_x
                self.joints[idx]["y"] = event.y * self.image_scale_y
                if hasattr(self, "original_frame"):
                    height, width = self.original_frame.shape[:2]
                    self._joint_reference_size = (float(width), float(height))
                self.draw_skeleton()
                name = self.joints[idx].get("name", f"kp_{idx}")
                self.status_var.set(f"Moved joint '{name}' to ({event.x:.0f}, {event.y:.0f}).")
                return

            name = simpledialog.askstring("New Keypoint", "Enter keypoint name:", parent=self.root)
            if name:
                x = event.x * self.image_scale_x
                y = event.y * self.image_scale_y
                self.joints.append({"name": name.strip(), "x": x, "y": y})
                self.update_keypoint_listbox(keep_selection=False)
                new_index = len(self.joints) - 1
                self._select_joint(new_index)
                self.status_var.set(f"Added joint '{name.strip()}'.")

    def on_canvas_right_click(self, event):
        self.adding_edge = False
        self.edge_start_joint = None
        self.draw_skeleton()
        self.status_var.set("Edge creation cancelled.")

    def get_joint_at(self, x, y):
        if self.image_scale_x == 0 or self.image_scale_y == 0:
            return None
        for i, joint in enumerate(self.joints):
            jx = joint["x"] / self.image_scale_x
            jy = joint["y"] / self.image_scale_y
            if abs(x - jx) < 5 and abs(y - jy) < 5:
                return i
        return None

    def add_edge(self):
        if len(self.joints) < 2:
            messagebox.showwarning("Add Edge", "Define at least two keypoints before adding edges.", parent=self.root)
            return
        self.adding_edge = True
        self.edge_start_joint = (
            self._active_joint_index if self._active_joint_index is not None else None
        )
        if self.edge_start_joint is not None:
            name = self.joints[self.edge_start_joint].get("name", f"kp_{self.edge_start_joint}")
            self.status_var.set(f"Connecting from '{name}'. Click another joint.")
        else:
            self.status_var.set("Click the first joint, then the joint to connect.")

    def remove_joint(self):
        selected_indices = self.keypoint_listbox.curselection()
        if selected_indices:
            index = selected_indices[0]
            del self.joints[index]
            adjusted_edges = []
            for start, end in self.edges:
                if start == index or end == index:
                    continue
                new_start = start - 1 if start > index else start
                new_end = end - 1 if end > index else end
                if new_start < len(self.joints) and new_end < len(self.joints):
                    adjusted_edges.append((new_start, new_end))
            self.edges = adjusted_edges
            self.update_keypoint_listbox(keep_selection=False)
            self.update_edge_listbox()
            self.draw_skeleton()

    def remove_edge(self):
        selected_indices = self.edge_listbox.curselection()
        if selected_indices:
            index = selected_indices[0]
            del self.edges[index]
            self.update_edge_listbox()
            self.draw_skeleton()

    def update_keypoint_listbox(self, keep_selection: bool = True):
        previous_selection = self.keypoint_listbox.curselection()
        self.keypoint_listbox.delete(0, tk.END)
        for joint in self.joints:
            self.keypoint_listbox.insert(tk.END, joint.get("name", ""))

        if self.joints and keep_selection and previous_selection:
            idx = min(previous_selection[0], len(self.joints) - 1)
            if idx >= 0:
                self._select_joint(idx)
                return
        elif not self.joints:
            self._active_joint_index = None
            self.status_var.set("No joints defined. Click in the canvas to add one.")
        self.draw_skeleton()

    def update_edge_listbox(self):
        self.edge_listbox.delete(0, tk.END)
        for edge in self.edges:
            start, end = edge
            if start >= len(self.joints) or end >= len(self.joints):
                continue
            name1 = self.joints[start].get("name", f"{start}")
            name2 = self.joints[end].get("name", f"{end}")
            self.edge_listbox.insert(tk.END, f"{name1} -> {name2}")

    def draw_skeleton(self):
        self.canvas.delete("skeleton")
        if not self.joints:
            return

        scale_x = self.image_scale_x or 1.0
        scale_y = self.image_scale_y or 1.0

        selection = self.keypoint_listbox.curselection()
        selected_idx = selection[0] if selection else None

        for idx, joint in enumerate(self.joints):
            x = joint["x"] / scale_x
            y = joint["y"] / scale_y
            radius = 5 if idx == selected_idx else 3
            fill = "#ffd966" if idx == selected_idx else "red"
            outline = "#444444" if idx == selected_idx else "white"
            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=fill, outline=outline, tags="skeleton")

        for start, end in self.edges:
            if start >= len(self.joints) or end >= len(self.joints):
                continue
            x1 = self.joints[start]["x"] / scale_x
            y1 = self.joints[start]["y"] / scale_y
            x2 = self.joints[end]["x"] / scale_x
            y2 = self.joints[end]["y"] / scale_y
            self.canvas.create_line(x1, y1, x2, y2, fill="white", width=2, tags="skeleton")

        if self.edge_start_joint is not None and self.edge_start_joint < len(self.joints):
            x = self.joints[self.edge_start_joint]["x"] / scale_x
            y = self.joints[self.edge_start_joint]["y"] / scale_y
            self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, outline="yellow", width=2, tags="skeleton")

    def save_skeleton(self):
        if not self.joints:
            messagebox.showwarning("Empty Skeleton", "Cannot save an empty skeleton.", parent=self.root)
            return

        path = filedialog.asksaveasfilename(
            title="Save Skeleton As",
            filetypes=[("JSON file", "*.json")],
            defaultextension=".json",
            parent=self.root
        )
        if not path:
            return

        keypoint_names = [
            str(joint.get("name", f"kp_{idx}")) for idx, joint in enumerate(self.joints)
        ]
        valid_edges = []
        named_edges = []
        for start, end in self.edges:
            if start < len(self.joints) and end < len(self.joints):
                valid_edges.append([int(start), int(end)])
                named_edges.append(
                    [keypoint_names[start], keypoint_names[end]]
                )

        skeleton_data = {
            "keypoints": keypoint_names,
            "keypoint_positions": [
                {
                    "name": joint.get("name", f"kp_{idx}"),
                    "x": float(joint.get("x", 0.0)),
                    "y": float(joint.get("y", 0.0)),
                }
                for idx, joint in enumerate(self.joints)
            ],
            "edges": named_edges,
            "index_edges": valid_edges,
        }
        if self._joint_reference_size:
            skeleton_data["frame_size"] = [
                float(self._joint_reference_size[0]),
                float(self._joint_reference_size[1]),
            ]

        try:
            Path(path).write_text(json.dumps(skeleton_data, indent=4), encoding="utf-8")
            # Sync saved edges back into the shared configuration so inference uses them.
            tuple_edges = [(int(a), int(b)) for a, b in valid_edges]
            if hasattr(self.app, "config") and hasattr(self.app.config, "pose_clustering"):
                self.app.config.pose_clustering.skeleton_connections = tuple_edges
                self.app.skeleton_connections = list(tuple_edges)
                try:
                    self.app.update_skeleton_dropdowns(force_refresh=True)
                except Exception:
                    pass
            self.app.log_message(f"Skeleton saved to {path} and connections synced.", "INFO")
            messagebox.showinfo("Skeleton Saved", f"Skeleton saved to {path} and connections synced to the UI.", parent=self.root)
            self.root.destroy()
        except Exception as e:
            self.app.log_message(f"Failed to save skeleton: {e}", "ERROR")
            messagebox.showerror("Save Error", f"Failed to save skeleton: {e}", parent=self.root)

