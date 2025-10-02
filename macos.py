
import AppKit
import fcntl

#load_framework(INCLUDE.AppKit)
#NSURL = autoclass('NSURL')
#NSOpenPanel = autoclass('NSOpenPanel')
#NSSavePanel = autoclass('NSSavePanel')
#NSOKButton = 1

class FileChooser:
    '''A native implementation of file chooser dialogs using Apple's API
    through pyobjus.

    Not implemented features:
    * filters (partial, wildcards are converted to extensions if possible.
        Pass the Mac-specific "use_extensions" if you can provide
        Mac OS X-compatible to avoid automatic conversion)
    * multiple (only for save dialog. Available in open dialog)
    * icon
    * preview
    '''

    mode = "open"
    path = None
    multiple = False
    filters = []
    preview = False
    title = None
    icon = None
    show_hidden = False
    use_extensions = False

    def __init__(self, *args, **kwargs):
        self._handle_selection = kwargs.pop(
            'on_selection', self._handle_selection
        )

        # Simulate Kivy's behavior
        for i in kwargs:
            setattr(self, i, kwargs[i])

    @staticmethod
    def _handle_selection(selection):
        '''
        Dummy placeholder for returning selection from chooser.
        '''
        return selection

    def run(self):
        panel = None
        if self.mode in ("open", "dir", "dir_and_files"):
            panel = AppKit.NSOpenPanel.openPanel()

            panel.setCanChooseDirectories_(self.mode != "open")
            panel.setCanChooseFiles_(self.mode != "dir")

            if self.multiple:
                panel.setAllowsMultipleSelection_(True)
        elif self.mode == "save":
            panel = AppKit.NSSavePanel.savePanel()
        else:
            assert False, self.mode

        panel.setCanCreateDirectories_(True)
        panel.setShowsHiddenFiles_(self.show_hidden)

        if self.title:
            panel.setTitle_(AppKit.NSString.alloc().initWithString_(self.title))

        # Mac OS X does not support wildcards unlike the other platforms.
        # This tries to convert wildcards to "extensions" when possible,
        # ans sets the panel to also allow other file types, just to be safe.
        if self.filters:
            filthies = []
            for f in self.filters:
                if isinstance(f, str):
                    f = (None, f)
                for s in f[1:]:
                    if not self.use_extensions:
                        if s.strip().endswith("*"):
                            continue
                    pystr = s.strip().split("*")[-1].split(".")[-1]
                    filthies.append(AppKit.NSString.alloc().initWithString_(pystr))

            ftypes_arr = AppKit.NSArray.alloc().initWithArray_(filthies)
            # todo: switch to allowedContentTypes
            panel.setAllowedFileTypes_(ftypes_arr)
            panel.setAllowsOtherFileTypes_(not self.use_extensions)

        if self.path:
            url = AppKit.NSURL.fileURLWithPath_(self.path)
            panel.setDirectoryURL_(url)

        selection = None

        if panel.runModal():
            if self.mode == "save" or not self.multiple:
                selection = [panel.filename().UTF8String()]
            else:
                filename = panel.filenames()
                selection = [
                    filename.objectAtIndex_(x).UTF8String()
                    for x in range(filename.count())]

        self._handle_selection(selection)

        return selection

def fadvice(file_path, use_cache=True):
    with open(file_path, "rb") as fd:
        # キャッシュの有効/無効を設定
        fcntl.fcntl(fd, fcntl.F_NOCACHE, 0 if use_cache else 1)
        
        # シーケンシャルアクセスの場合、先読みを有効化
        fcntl.fcntl(fd, 45, 1 if use_cache else 0)  # F_RDAHEAD の値は macOS でのみ有効
