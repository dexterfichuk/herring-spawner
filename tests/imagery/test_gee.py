from herring_spawner.imagery.gee import GeeSentinel2Provider


def test_gee_provider_uses_redd_fish_project_by_default():
    provider = GeeSentinel2Provider()

    assert provider.project == "redd-fish"
    assert provider.collection == "COPERNICUS/S2_SR_HARMONIZED"
    assert provider.cloud_collection == "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"


def test_scene_search_request_is_provider_neutral():
    provider = GeeSentinel2Provider(project="custom-project")
    request = provider.build_search_request(
        bounds=(-126.3, 50.7, -126.1, 50.9),
        start_date="2026-03-25",
        end_date="2026-04-14",
        max_cloud=50,
    )

    assert request["provider"] == "gee"
    assert request["project"] == "custom-project"
    assert request["collection"] == "COPERNICUS/S2_SR_HARMONIZED"
    assert request["bounds"] == (-126.3, 50.7, -126.1, 50.9)
