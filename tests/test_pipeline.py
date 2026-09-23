import cv2
import numpy as np
import pytest

from src.blending import gain_compensation, largest_valid_rectangle, multiband_blend, feather_blend, seam_labels
from src.feature import detect_features, match_features
from src.homography import estimate_transform, focals_from_homography, centered, transform_is_plausible
from src.image_io import decode_image, encode_image
from src.stitcher import StitchError, StitchSettings, stitch
from src.synthetic import make_planar_set, make_rotation_set, make_scene
from src.warping import WarpedImage


def corner_error(result, synthetic, order):
    """ความคลาดเคลื่อนเฉลี่ย (px) ของมุมภาพเมื่อเทียบกับ homography จริง"""
    reference = result.reference
    errors = []
    for index in result.included:
        if index == reference:
            continue
        truth = np.linalg.inv(synthetic.view_to_scene[order[reference]]) @ synthetic.view_to_scene[order[index]]
        height, width = result.images[index].shape[:2]
        corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
        difference = cv2.perspectiveTransform(corners, result.transforms[index]) - cv2.perspectiveTransform(corners, truth)
        errors.append(np.linalg.norm(difference, axis=2).mean())
    return float(np.mean(errors))


@pytest.fixture(scope="module")
def planar_pair():
    return make_planar_set(count=2, seed=11)


def test_decode_rejects_non_image_and_resizes_large_image():
    with pytest.raises(ValueError):
        decode_image(b"not an image")
    image = np.zeros((900, 3000, 3), np.uint8)
    decoded, original = decode_image(encode_image(image, ".png"), max_side=1200)
    assert original == (3000, 900)
    assert max(decoded.shape[:2]) == 1200


@pytest.mark.parametrize("detector", ["SIFT", "ORB"])
def test_feature_matching_finds_ratio_test_matches(planar_pair, detector):
    first = detect_features(planar_pair.images[0], detector, 3000)
    second = detect_features(planar_pair.images[1], detector, 3000)
    matches = match_features(first, second, ratio=0.75)
    assert len(first) > 300 and len(second) > 300
    assert len(matches) > 50
    # ไม่มีจุดในภาพที่สองถูกใช้ซ้ำ
    assert len({match.trainIdx for match in matches}) == len(matches)


def test_blank_image_has_no_matches_instead_of_crashing():
    blank = detect_features(np.full((300, 400, 3), 128, np.uint8))
    textured = detect_features(make_scene(400, 300, seed=3))
    assert match_features(blank, textured) == []
    assert match_features(textured, blank) == []


def test_ransac_recovers_homography_despite_outliers():
    rng = np.random.default_rng(0)
    truth = np.array([[1.02, 0.03, 180.0], [-0.02, 0.99, 12.0], [2e-5, -1e-5, 1.0]])
    source = rng.uniform(0, [640, 480], size=(300, 2)).astype(np.float32)
    destination = cv2.perspectiveTransform(source.reshape(-1, 1, 2), truth).reshape(-1, 2)
    destination += rng.normal(0, 0.5, destination.shape).astype(np.float32)
    outliers = rng.choice(300, 90, replace=False)
    destination[outliers] = rng.uniform(0, [900, 480], size=(90, 2))

    estimate = estimate_transform(source, destination, threshold=3.0, source_size=(640, 480))
    assert estimate.ok
    assert estimate.num_inliers >= 200
    assert not estimate.inlier_mask[outliers].any()
    np.testing.assert_allclose(estimate.matrix / estimate.matrix[2, 2], truth, rtol=0.02, atol=0.5)


def test_verification_rejects_random_correspondences():
    rng = np.random.default_rng(1)
    source = rng.uniform(0, 640, size=(60, 2))
    destination = rng.uniform(0, 640, size=(60, 2))
    estimate = estimate_transform(source, destination, source_size=(640, 480))
    assert not estimate.ok
    assert estimate.failure_reason


def test_too_few_points_is_reported():
    estimate = estimate_transform(np.zeros((3, 2)), np.zeros((3, 2)))
    assert not estimate.ok
    assert "อย่างน้อย 4" in estimate.failure_reason


def test_plausibility_rejects_reflection_horizon_and_extreme_scale():
    assert transform_is_plausible(np.eye(3), 640, 480)[0]
    for matrix in [
        np.array([[-1.0, 0, 640], [0, 1, 0], [0, 0, 1]]),  # กลับด้าน
        np.array([[1.0, 0, 0], [0, 1, 0], [-0.002, 0, 1]]),  # มุมขวาเลยเส้นขอบฟ้า
        np.diag([12.0, 12.0, 1.0]),  # ขยายผิดปกติ
        np.full((3, 3), np.nan),
    ]:
        assert not transform_is_plausible(matrix, 640, 480)[0]


def test_focal_length_from_rotation_homography():
    focal = 700.0
    camera = np.array([[focal, 0, 0], [0, focal, 0], [0, 0, 1]])
    yaw, pitch = np.radians(25), np.radians(3)
    rotation = cv2.Rodrigues(np.array([pitch, yaw, 0.0]))[0]
    homography = camera @ rotation @ np.linalg.inv(camera)
    first, second = focals_from_homography(homography)
    assert first == pytest.approx(focal, rel=0.02)
    assert second == pytest.approx(focal, rel=0.02)
    # centered() ไม่เปลี่ยนเมทริกซ์ถ้าเลื่อนจุดศูนย์กลางไปแล้วกลับมา
    shifted = centered(np.eye(3), (640, 480), (640, 480))
    np.testing.assert_allclose(shifted, np.eye(3))


def test_two_images_in_reverse_order_are_stitched_accurately():
    synthetic = make_planar_set(count=2, seed=21)
    order = [1, 0]  # อัปโหลดภาพขวาก่อน (โค้ดเวอร์ชันแรกตัดภาพซ้ายทิ้ง)
    result = stitch([synthetic.images[index] for index in order], StitchSettings(crop=False))
    assert result.projection == "planar"
    assert result.included == [1, 0]
    assert corner_error(result, synthetic, order) < 1.5
    width = result.panorama.shape[1]
    assert width > 1.3 * synthetic.images[0].shape[1]


def test_shuffled_multi_image_set_is_ordered_and_seamless():
    synthetic = make_planar_set(count=5, seed=22, gain_range=0.2, vignetting=0.3)
    order = [3, 0, 4, 1, 2]
    result = stitch([synthetic.images[index] for index in order], StitchSettings())
    assert [order[index] for index in result.included] == [0, 1, 2, 3, 4]
    assert corner_error(result, synthetic, order) < 1.5
    assert result.gains is not None
    assert sum(pair.in_tree for pair in result.pairs) == 4
    # หลัง auto-crop ต้องไม่มีขอบดำ
    assert (result.panorama.reshape(-1, 3).max(axis=1) > 0).mean() > 0.999


def test_unrelated_image_is_excluded_with_warning():
    synthetic = make_planar_set(count=3, seed=23)
    unrelated = make_planar_set(count=1, seed=99).images[0]
    result = stitch(synthetic.images + [unrelated], StitchSettings())
    assert sorted(result.included) == [0, 1, 2]
    assert result.excluded == [3]
    assert any("ภาพ 4" in warning for warning in result.warnings)


def test_images_without_overlap_raise_readable_error():
    first = make_planar_set(count=1, seed=31).images[0]
    second = make_planar_set(count=1, seed=32).images[0]
    with pytest.raises(StitchError) as error:
        stitch([first, second])
    assert "ซ้อนทับ" in str(error.value)
    assert len(error.value.pairs) == 1


def test_wide_rotation_switches_to_cylindrical_and_estimates_focal():
    synthetic = make_rotation_set(count=5, yaw_step=30, seed=41)
    result = stitch(synthetic.images, StitchSettings())
    assert result.projection == "cylindrical"
    assert sorted(result.included) == list(range(5))
    assert result.focal_estimated
    assert result.focal == pytest.approx(synthetic.focal, rel=0.05)
    with pytest.raises(StitchError):
        stitch(synthetic.images, StitchSettings(projection="planar"))


def test_orb_pipeline_also_stitches():
    synthetic = make_planar_set(count=3, seed=51)
    result = stitch(synthetic.images, StitchSettings(detector="ORB", crop=False))
    assert sorted(result.included) == [0, 1, 2]
    assert corner_error(result, synthetic, [0, 1, 2]) < 5.0  # ORB ไม่มี sub-pixel refinement แบบ SIFT


def _two_flat_warps(left_value, right_value):
    left = WarpedImage(0, np.full((100, 160, 3), left_value, np.uint8), np.full((100, 160), 255, np.uint8), 0, 0)
    right = WarpedImage(1, np.full((100, 160, 3), right_value, np.uint8), np.full((100, 160), 255, np.uint8), 100, 0)
    return [left, right]


def test_gain_compensation_equalizes_overlap_brightness():
    warps = _two_flat_warps(100, 140)
    gains = gain_compensation(warps)
    assert gains[0, 0] * 100 == pytest.approx(gains[1, 0] * 140, rel=0.05)


@pytest.mark.parametrize("blend", [feather_blend, multiband_blend])
def test_blending_keeps_colors_away_from_seam_and_is_smooth(blend):
    warps = _two_flat_warps(60, 200)
    panorama, mask = blend(warps, (260, 100))
    assert mask.all()
    assert abs(int(panorama[50, 5, 0]) - 60) <= 2
    assert abs(int(panorama[50, 255, 0]) - 200) <= 2
    # ไม่มีขั้นกระโดดของความสว่างระหว่างพิกเซลติดกัน (รอยต่อแข็ง 140 ระดับ)
    assert np.abs(np.diff(panorama[50, :, 0].astype(int))).max() < 40


def test_seam_is_placed_inside_overlap():
    warps = _two_flat_warps(0, 255)
    labels = seam_labels(warps, (260, 100))
    assert (labels[:, :100] == 0).all() and (labels[:, 160:] == 1).all()
    boundary = np.argmax(labels[50] == 1)
    assert 115 <= boundary <= 145


def test_largest_valid_rectangle_excludes_black_border():
    mask = np.zeros((400, 900), np.uint8)
    cv2.fillPoly(mask, [np.array([[40, 30], [860, 60], [880, 370], [20, 390]])], 255)
    x, y, width, height = largest_valid_rectangle(mask)
    assert mask[y:y + height, x:x + width].all()
    assert width * height > 0.8 * np.count_nonzero(mask)
