import qiime2.plugin.model as model

class ModmagsFormat(model.BinaryFileFormat):

    def validate(self, *args):
        pass

# Define a directory format for the CSR Matrix Modmags
ModmagsDirFormat = model.SingleFileDirectoryFormat(
    'ModmagsDirFormat', 'modmags.npz', ModmagsFormat)