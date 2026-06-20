import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DRAGGABLE_PATH = PROJECT_ROOT / "widgets" / "draggable_widget.py"


def _load_class_function(class_name, function_name):
    source = DRAGGABLE_PATH.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return ast.get_source_segment(source, child)
    raise AssertionError(f"{class_name}.{function_name} was not found")


def _load_function(function_name):
    source = DRAGGABLE_PATH.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"{function_name} was not found")


class DragToTrashFlowTest(unittest.TestCase):
    def test_drag_items_publish_finder_compatible_file_types(self):
        source = DRAGGABLE_PATH.read_text()
        make_item_source = _load_function("_make_file_pasteboard_item")
        start_source = _load_class_function("DraggableWidget", "start_drag")

        self.assertIn("NSPasteboardItem", source)
        self.assertIn("NSPasteboardTypeFileURL", source)
        self.assertIn("NSFilenamesPboardType", source)
        self.assertIn("NSURLPboardType", source)
        self.assertIn("item.setString_forType_(url_string, NSPasteboardTypeFileURL)", make_item_source)
        self.assertIn("item.setString_forType_(url_string, NSURLPboardType)", make_item_source)
        self.assertIn("item.setPropertyList_forType_([path], NSFilenamesPboardType)", make_item_source)
        self.assertIn("_make_file_pasteboard_item(file_path)", start_source)

    def test_drag_source_allows_copy_and_delete_operations(self):
        source = DRAGGABLE_PATH.read_text()
        mask_source = _load_class_function(
            "_DraggingSource", "draggingSession_sourceOperationMaskForDraggingContext_"
        )
        end_source = _load_class_function(
            "_DraggingSource", "draggingSession_endedAtPoint_operation_"
        )
        start_source = _load_class_function("DraggableWidget", "start_drag")

        self.assertIn("NSDragOperationDelete, NSWorkspace, NSObject", source)
        self.assertIn("class _DraggingSource(NSObject):", source)
        self.assertIn("NSDragOperationDelete", source)
        self.assertIn("return NSDragOperationCopy | NSDragOperationDelete", mask_source)
        self.assertIn("owner._on_dragging_session_ended(operation)", end_source)
        self.assertIn("self._current_drag_file_paths", start_source)
        self.assertIn("_DraggingSource.alloc().init()", start_source)
        self.assertIn("self._drag_source.owner = self", start_source)
        self.assertIn("beginDraggingSessionWithItems_event_source_", start_source)
        self.assertIn("self._drag_source\n        )", start_source)
        self.assertNotIn("main_window\n        )", start_source)

    def test_drag_end_recycles_only_delete_operation(self):
        end_source = _load_class_function("DraggableWidget", "_on_dragging_session_ended")
        delete_source = _load_class_function("DraggableWidget", "_drag_operation_requests_delete")
        recycle_source = _load_class_function("DraggableWidget", "_recycle_drag_files")

        self.assertIn("operation == NSDragOperationNone", end_source)
        self.assertIn("self._drag_operation_requests_delete(operation)", end_source)
        self.assertIn("self._recycle_drag_files(self._current_drag_file_paths)", end_source)
        self.assertIn("self._current_drag_file_paths = []", end_source)
        self.assertIn("self._drag_source = None", end_source)
        self.assertIn("int(operation) & int(NSDragOperationDelete)", delete_source)
        self.assertIn("os.path.isfile(path)", recycle_source)
        self.assertIn("NSWorkspace.sharedWorkspace().recycleURLs_completionHandler_", recycle_source)


if __name__ == "__main__":
    unittest.main()
