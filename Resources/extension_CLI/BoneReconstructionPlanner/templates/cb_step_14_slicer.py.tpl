import qt

# Change layout to BoneReconstructionPlanner (custom BRP layout)
# This layout shows: [3D Mandible View | Red Slice (Axial)] on top, [3D Fibula View] on bottom.

# Define layout constants (as used by the BoneReconstructionPlanner extension)
MANDIBLE_VIEW_SINGLETON_TAG = "1"
FIBULA_VIEW_SINGLETON_TAG = "2"
BRP_LAYOUT_ID = 101

# Register the custom layout description if not already registered
layoutManager = slicer.app.layoutManager()
layoutNode = layoutManager.layoutLogic().GetLayoutNode()

# Check if the layout ID is already registered
layoutExists = any(
    action.data() == BRP_LAYOUT_ID
    for action in slicer.util.mainWindow().findChild('QToolBar', 'ViewToolBar')
    .widgetForAction(slicer.util.mainWindow().findChild('QToolBar', 'ViewToolBar').actions()[0])
    .menu().actions()
    if action.data() is not None
)

if not layoutExists:
    brpLayoutXml = """\
    <layout type="vertical">
    <item>
      <layout type="horizontal">
      <item>
        <view class="vtkMRMLViewNode" singletontag="{mandibleTag}">
        <property name="viewlabel" action="default">1</property>
        </view>
      </item>
      <item>
        <view class="vtkMRMLSliceNode" singletontag="Red">
        <property name="orientation" action="default">Axial</property>
        <property name="viewlabel" action="default">R</property>
        <property name="viewcolor" action="default">#F34A33</property>
        </view>
      </item>
      </layout>
    </item>
    <item>
      <view class="vtkMRMLViewNode" singletontag="{fibulaTag}">
      <property name="viewlabel" action="default">2</property>
      </view>
    </item>
    </layout>""".format(mandibleTag=MANDIBLE_VIEW_SINGLETON_TAG, fibulaTag=FIBULA_VIEW_SINGLETON_TAG)

    layoutNode.AddLayoutDescription(BRP_LAYOUT_ID, brpLayoutXml)

    # Also add a menu entry for the layout selector toolbar
    viewToolBar = slicer.util.mainWindow().findChild('QToolBar', 'ViewToolBar')
    layoutMenu = viewToolBar.widgetForAction(viewToolBar.actions()[0]).menu()
    layoutSwitchAction = layoutMenu.addAction("BoneReconstructionPlanner")
    layoutSwitchAction.setData(BRP_LAYOUT_ID)
    layoutSwitchAction.setToolTip('3D Mandible View, Red Slice and 3D Fibula View')

# Switch to the BoneReconstructionPlanner layout
layoutManager.setLayout(BRP_LAYOUT_ID)